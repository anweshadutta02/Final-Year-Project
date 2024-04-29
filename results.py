import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import unicodedata

# File Paths
udemy_courses = "udemy_courses.parquet"
new_ratings = "new_ratings.parquet"
users = "Users.parquet"

def clean_text(text):
    """
    Clean and normalize the text by removing special characters, accents, and converting to lowercase.
    """
    if not pd.isnull(text):
        text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")
        return text.lower().strip()
    return text

# Load the data for content-based recommendations
def load_udemy_courses():
    courses_df = pd.read_parquet(udemy_courses)
    tf_idf = TfidfVectorizer(stop_words="english")
    tf_idf_matrix = tf_idf.fit_transform(courses_df["course_title"])
    cosine_sim = cosine_similarity(tf_idf_matrix)
    course_title_to_index = {title: i for i, title in enumerate(courses_df["course_title"])}
    return courses_df, cosine_sim, course_title_to_index

courses_df, cosine_sim, course_title_to_index = load_udemy_courses()

# Load the data for collaborative filtering recommendations
def load_users_and_ratings():
    users_df = pd.read_parquet(users)
    ratings_df = pd.read_parquet(new_ratings)
    df2 = pd.merge(ratings_df, users_df, on='User-ID')
    df2 = pd.merge(df2, courses_df, on='course_id')
    user_course_matrix = df2.pivot_table(index='User-ID', columns='course_id', values='Rating').fillna(0)
    course_similarity_df = pd.DataFrame(cosine_similarity(user_course_matrix.T), index=user_course_matrix.columns, columns=user_course_matrix.columns)
    return df2, user_course_matrix, course_similarity_df

df2, user_course_matrix, course_similarity_df = load_users_and_ratings()

# Recommenders
def content_based_recommender(course_title):
    course_index = course_title_to_index.get(course_title)
    if course_index is None:
        return ["Course not found."]
    similarity_scores = pd.DataFrame(cosine_sim[course_index], columns=["score"])
    course_indices = similarity_scores.sort_values("score", ascending=False)[1:7].index
    filtered_indices = [idx for idx in course_indices if idx != course_index and idx < len(courses_df)]
    recommended_courses = [courses_df["course_title"].iloc[idx] for idx in filtered_indices]
    return recommended_courses[:5]

def collaborative_recommender(course_title):
    course_title = course_title.strip().lower()
    if course_title not in courses_df['course_title'].str.lower().unique():
        return f"Course '{course_title}' not present in 'course_title' column in the dataset."
    course_id = courses_df[courses_df['course_title'].str.lower() == course_title]['course_id'].iloc[0]
    similar_courses = course_similarity_df[course_id]
    similar_courses = similar_courses.sort_values(ascending=False)
    recommended_course_ids = similar_courses.head(11).index.tolist()
    recommended_course_ids.remove(course_id) 
    recommended_courses = [courses_df[courses_df['course_id'] == cid]['course_title'].iloc[0] for cid in recommended_course_ids if cid in courses_df['course_id'].values]
    return recommended_courses

def hybrid_recommender(course_title):
    content_based_rec = content_based_recommender(course_title)[:4]
    collaborative_filtering_rec = collaborative_recommender(course_title)
    
    # Ensure both are lists
    if not isinstance(content_based_rec, list):
        content_based_rec = [content_based_rec]
    if not isinstance(collaborative_filtering_rec, list):
        collaborative_filtering_rec = [collaborative_filtering_rec]
    
    features = ['Content-Based'] * len(content_based_rec) + ['Collaborative'] * len(collaborative_filtering_rec)
    label_encoder = LabelEncoder()
    X_encoded = label_encoder.fit_transform(features)
    y = [1] * len(content_based_rec) + [0] * len(collaborative_filtering_rec)
    model = LogisticRegression()
    model.fit(X_encoded.reshape(-1, 1), y)
    all_features = ['Content-Based'] * 6 + ['Collaborative'] * 6
    all_features_encoded = label_encoder.transform(all_features)
    all_probs = model.predict_proba(all_features_encoded.reshape(-1, 1))[:, 1]
    rec_with_probs = list(zip(all_features, all_probs, content_based_rec + collaborative_filtering_rec))
    rec_with_probs.sort(key=lambda x: x[1], reverse=True)
    hybrid_rec = [item[2] for item in rec_with_probs[:6]]
    return hybrid_rec

# def evaluate_model_recommendations(model_recommender, course_name):
#     ratings_array = []
#     non_zero_ratings_for_course = df2[(df2['course_title'] == course_name) & (df2['Rating'] != 0)]['Rating']
#     mean_rating_course = np.mean(non_zero_ratings_for_course)
#     ratings_array.append(mean_rating_course)

#     recommended_courses = model_recommender(course_name)
#     ratings_array_2 = [mean_rating_course] * len(recommended_courses)
#     ratings_for_each_course = []
#     for recommended_course in recommended_courses:
#         non_zero_ratings_for_course = df2[(df2['course_title'] == recommended_course) & (df2['Rating'] != 0)]['Rating']
#         mean_rating = np.mean(non_zero_ratings_for_course)
#         ratings_for_each_course.append(mean_rating)
#     mae = mean_absolute_error(ratings_array_2, ratings_for_each_course)
#     mse = mean_squared_error(ratings_array_2, ratings_for_each_course)

#     return {"MAE": mae, "MSE": mse}

# # Evaluate the models
# user_input = 'Build Websites from Scratch with HTML & CSS'
# course_name = next((col for col in courses_df['course_title'] if user_input in col), None)

# if course_name is not None:
#     evaluation_results = {
#         "Content-Based": evaluate_model_recommendations(content_based_recommender, course_name),
#         "Collaborative": evaluate_model_recommendations(collaborative_recommender, course_name),
#         "Hybrid": evaluate_model_recommendations(hybrid_recommender, course_name)
#     }

#     # Print the evaluation results
#     print("Evaluation Results for", course_name)
#     for model, metrics in evaluation_results.items():
#         print(f"{model} - MAE: {metrics['MAE']}, MSE: {metrics['MSE']}")
# else:
#     print("Course not found in dataset.")

# Function to evaluate models on multiple courses
def evaluate_models_on_multiple_courses(model_recommender, num_courses=800):
    random_courses = []

    subjects = ['Business Finance', 'Graphic Design', 'Musical Instruments', 'Web Development']

    for subject in subjects:
        random_courses += courses_df[courses_df['subject'] == subject]['course_title'].sample(200).tolist()

    avg_mae = 0
    avg_mse = 0
    for course_name in random_courses:
        ratings_array = []
        non_zero_ratings_for_course = df2[(df2['course_title'] == course_name) & (df2['Rating'] != 0)]['Rating']
        mean_rating_course = np.mean(non_zero_ratings_for_course)
        ratings_array.append(mean_rating_course)

        recommended_courses = model_recommender(course_name)
        ratings_array_2 = [mean_rating_course] * len(recommended_courses)
        ratings_for_each_course = []
        for recommended_course in recommended_courses:
            non_zero_ratings_for_course = df2[(df2['course_title'] == recommended_course) & (df2['Rating'] != 0)]['Rating']
            mean_rating = np.mean(non_zero_ratings_for_course)
            ratings_for_each_course.append(mean_rating)

        # Replace NaN values with the mean rating of the course
        ratings_array_2 = np.nan_to_num(ratings_array_2, nan=mean_rating_course)
        ratings_for_each_course = np.nan_to_num(ratings_for_each_course, nan=mean_rating_course)

        mae = mean_absolute_error(ratings_array_2, ratings_for_each_course)
        mse = mean_squared_error(ratings_array_2, ratings_for_each_course)

        avg_mae += mae
        avg_mse += mse

    avg_mae /= num_courses
    avg_mse /= num_courses
    print("done")

    return {"Average MAE": avg_mae, "Average MSE": avg_mse}

# Evaluate the models on multiple courses
evaluation_results = {
    "Content-Based": evaluate_models_on_multiple_courses(content_based_recommender),
    "Collaborative": evaluate_models_on_multiple_courses(collaborative_recommender),
    "Hybrid": evaluate_models_on_multiple_courses(hybrid_recommender)
}

# Print the evaluation results
print("Average Evaluation Results for Multiple Courses:")
for model, metrics in evaluation_results.items():
    print(f"{model} - Average MSE: {metrics['Average MSE']}")
