from flask import Flask, jsonify, request
import pandas as pd
from transformers import pipeline
from collections import Counter
from openai import OpenAI
import warnings
import os

app = Flask(__name__)

file_path = "./data.csv"
warnings.filterwarnings('ignore')
test_data = pd.read_csv(file_path, encoding='unicode_escape')

openai_api_key = os.environ.get("OPENAI_API_KEY")
print(openai_api_key)
if not openai_api_key:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

    
def generate_summary_recommendation(review, aspect, sentiment):
    prompt = (
        f"Based on the review text, aspect, and sentiment provided, generate a concise summary recommendation that are logical and reflect clear cause and effect. "
        f"When a review expresses a negative sentiment, begin the recommendation with the word 'Avoid.' Highlight the main criticism mentioned in the review and propose a specific action to address this issue. For example, if the criticism pertains to slow service, the recommendation could be formulated as: 'Avoid the slow service by hiring additional staff to enhance efficiency.' "
        f"Whenever a review expresses positive sentiment, start with 'You should provide,' emphasize the praised aspects, and suggest expanding them, such as 'You should provide more spacious parking spaces for cars.'"
        f"Ensure the recommendation is directly linked to the aspect and sentiment, and is limited to one sentence of 5-10 words. "
        f"Avoid contradictions between negative and positive recommendations; for example, 'Avoid the drinks' should not coincide with 'You should provide a variety of delicious drinks'. "
        f"When generating recommendations based on customer feedback, ensure that no staff names are mentioned. Focus on the qualities of the service or product itself. For example, rather than citing a specific staff member, the recommendation could be: 'You should provide consistently friendly service, similar to the best examples observed.' This approach maintains privacy and emphasizes service quality."
        f"Exclude reviews that are unclear, lack meaningful content, or contain vague opinions. "
        f"When customers consistently report a specific issue with a product, like the bitterness of a drink, provide a targeted recommendation to address the problem. For instance, if the Spanish Latte is frequently described as too bitter, the recommendation could be structured as: 'Avoid making the Spanish Latte drinks bitter by carefully balancing the ingredients to enhance overall flavor.'"
        f"When generating recommendations based on customer feedback, differentiate clearly between staff-related issues and service-related issues. For staff issues, focus on training and personnel management, such as 'Provide consistent and friendly service training for all staff.' For service issues, address operational improvements, like 'Improve service speed and customer interaction by optimizing workflows.' Ensure that the recommendations for each aspect are distinct and tailored to address specific concerns without overlap."
        f"When generating recommendations based on customer feedback, ensure that each recommendation directly addresses the specific issues mentioned. Maintain a focus on the relevant context, and avoid mixing different matters or contexts in the same sentence. This ensures that the advice is directly pertinent and effectively resolves the concerns raised without introducing unrelated topics."
        f"Here's the input for your reference:\nReview: {review}\nAspect: {aspect}\nSentiment: {sentiment}"
    )
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=300,
    )
    return response.choices[0].message.content

def get_recommendations(data, location_id):
    location_data = data[data['Store_locations_ID'] == location_id]
    location_data['aspect_sentiment_pairs'] = location_data.apply(
        lambda row: list(zip(row['aspectCategory'].split(', '), row['sentiment'].split(', '))),
        axis=1
    )
    recommendations = []
    all_pairs = [pair for sublist in location_data['aspect_sentiment_pairs'] for pair in sublist]
    all_aspects = [aspect for aspect, sentiment in all_pairs]
    all_sentiments = [sentiment for aspect, sentiment in all_pairs]
    aspect_counts = pd.Series(all_aspects).value_counts().nlargest(3)

    for aspect_category in aspect_counts.index:
        aspect_reviews = location_data[location_data['aspect_sentiment_pairs'].apply(
            lambda pairs: any(aspect == aspect_category for aspect, sentiment in pairs)
        )]
        sentiment_recommendations = {}
        for sentiment in set(all_sentiments):
            sentiment_reviews = aspect_reviews[aspect_reviews['aspect_sentiment_pairs'].apply(
                lambda pairs: any(aspect == aspect_category and sentiment == sent for aspect, sent in pairs)
            )]
            top_sentiment_reviews = sentiment_reviews.head(3)['Text'].tolist()

            if top_sentiment_reviews:
                merged_sentiment_summary = " and ".join(top_sentiment_reviews)
                sentiment_recommendation = generate_summary_recommendation(merged_sentiment_summary, aspect_category, sentiment)
                sentiment_recommendations[sentiment] = {
                    'Top Reviews': top_sentiment_reviews,
                    'Summary': merged_sentiment_summary,
                    'Recommendation': sentiment_recommendation
                }
        recommendations.append({
            'Aspect Category': aspect_category,
            **sentiment_recommendations
        })
    return recommendations

def display_recommendations_summary_df(recommendations):
    things_to_consider = []
    things_to_avoid = []
    for recommendation in recommendations:
        if 'positive' in recommendation:
            things_to_consider.append(recommendation['positive']['Recommendation'])
        if 'negative' in recommendation:
            things_to_avoid.append(recommendation['negative']['Recommendation'])
    max_length = max(len(things_to_consider), len(things_to_avoid))
    things_to_consider.extend([''] * (max_length - len(things_to_consider)))
    things_to_avoid.extend([''] * (max_length - len(things_to_avoid)))
    data = {
        "Things to Consider": things_to_consider,
        "Things to Avoid": things_to_avoid
    }
    df = pd.DataFrame(data)
    return df

@app.route('/display_recommendations_summary_df/<int:location_id>', methods=['GET'])
def get_recommendations_summary_df(location_id):
    recommendations = get_recommendations(test_data, location_id)
    df = display_recommendations_summary_df(recommendations)
    return jsonify(df.to_dict())
if __name__ == '__main__':
    app.run(debug=True)

