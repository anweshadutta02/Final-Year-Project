import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pyarrow.parquet import ParquetDataset

# File Paths
udemy_courses = "Files/udemy_courses.csv"
new_ratings = "Files/new_ratings.csv"
users = "Files/Users.csv"

# Load the datas for content-based recommendations
@st.cache_resource
def load_udemy_courses():
    # dataset = ParquetDataset(udemy_courses)
    df = pd.read_csv(udemy_courses)
    tf_idf = TfidfVectorizer(stop_words="english")
    tf_idf_matrix = tf_idf.fit_transform(df["course_title"])
    cosine_sim = cosine_similarity(tf_idf_matrix)
    course_title_to_index = {title: i for i, title in enumerate(df["course_title"])}
    return df, cosine_sim, course_title_to_index

df, cosine_sim, course_title_to_index = load_udemy_courses()

# Load the datas for collaborative filtering recommendations
@st.cache_resource
def load_users_and_ratings():
    # users_dataset = ParquetDataset(users)
    users_df = pd.read_csv(users)
    # ratings_dataset = ParquetDataset(new_ratings)
    ratings_df = pd.read_csv(new_ratings)
    df2 = pd.merge(ratings_df, users_df, on='User-ID')
    df2 = pd.merge(df2, df, on='course_id')
    user_course_matrix = df2.pivot_table(index='User-ID', columns='course_id', values='Rating').fillna(0)
    course_similarity_df = pd.DataFrame(cosine_similarity(user_course_matrix.T), index=user_course_matrix.columns, columns=user_course_matrix.columns)
    return user_course_matrix, course_similarity_df

user_course_matrix, course_similarity_df = load_users_and_ratings()

# Recommenders
def content_based_recommender(title, df, cosine_sim, course_title_to_index):
    course_index = course_title_to_index.get(title)
    if course_index is None:
        return ["Course not found."]
    similarity_scores = pd.DataFrame(cosine_sim[course_index], columns=["score"])
    course_indices = similarity_scores.sort_values("score", ascending=False)[1:6].index
    recommended_courses = [df["course_title"].iloc[idx] for idx in course_indices if idx < len(df)]
    return recommended_courses

def collaborative_recommender(course_name, user_course_matrix, course_similarity_df, df, num_recommendations=10):
    # Clean and preprocess course name (remove leading/trailing whitespaces, make it lowercase)
    course_name = course_name.strip().lower()

    # Check if the course name exists in the 'course_title' column of the original DataFrame 'df'
    if course_name not in df['course_title'].str.lower().unique():
        return f"Course '{course_name}' not present in 'course_title' column in the dataset."

    # Find the corresponding course ID from 'course_id' column
    course_id = df[df['course_title'].str.lower() == course_name]['course_id'].iloc[0]

    similar_courses = course_similarity_df[course_id]
    similar_courses = similar_courses.sort_values(ascending=False)
    recommended_course_ids = similar_courses.head(num_recommendations + 1).index.tolist()
    recommended_course_ids.remove(course_id)  # Remove the input course itself
    recommended_courses = [df[df['course_id'] == cid]['course_title'].iloc[0] for cid in recommended_course_ids if cid in df['course_id'].values]
    return recommended_courses

def hybrid_recommender(course_name, content_weight=0.5):
    # Get recommendations from both models
    content_based_rec = content_based_recommender(course_name, df, cosine_sim, course_title_to_index)
    collaborative_filtering_rec = collaborative_recommender(course_name, user_course_matrix, course_similarity_df, df, num_recommendations=7)
    
    # Calculate hybrid recommendations as a weighted average
    hybrid_rec = []
    for item in content_based_rec:
        hybrid_rec.append((item, content_weight))
    
    for item in collaborative_filtering_rec:
        hybrid_rec.append((item, 1 - content_weight))

    # Sort recommendations by the weighted score
    hybrid_rec.sort(key=lambda x: x[1], reverse=True)

    # Extract recommended items without weights
    hybrid_rec = [item[0] for item in hybrid_rec][:5]

    return hybrid_rec 

# Streamlit app
st.title("Udemy Course Recommender")
user_input = st.text_input("Enter a subject or course title:")

# Create buttons for tabs
col1, col2, col3 = st.columns(3)
content_based_button = col1.button("Content-Based")
collaborative_button = col2.button("Collaborative")
hybrid_button = col3.button("Hybrid")

# Update selected tab index based on button clicks
if content_based_button:
    selected_tab_index = 0
elif collaborative_button:
    selected_tab_index = 1
elif hybrid_button:
    selected_tab_index = 2
else:
    selected_tab_index = None  # No tab is selected by default

if user_input:
    course = next((col for col in df['course_title'] if user_input in col), None)
    if course:
        if selected_tab_index is None:
            st.write("Click on any button to show different recommendations")  # Show this message only if no button has been clicked
    else:
        st.write("Course not found.")

    # Display content based on selected tab
    if selected_tab_index is not None:  # Only show recommendations if a button has been clicked
        if selected_tab_index == 0:
            recommendations = content_based_recommender(course, df, cosine_sim, course_title_to_index)
        elif selected_tab_index == 1:
            recommendations = collaborative_recommender(course, user_course_matrix, course_similarity_df, df, num_recommendations=5)
        else:
            recommendations = hybrid_recommender(course)

        if isinstance(recommendations, str):
            st.write(recommendations)
        else:
            st.write("Here are some recommended courses:")
            for rec in recommendations:
                st.write(f"- {rec}")
