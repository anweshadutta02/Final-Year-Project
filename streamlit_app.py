import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(page_title="Udemy Course Recommender", layout="wide")

# File Paths
udemy_courses = "udemy_courses.parquet"
new_ratings = "new_ratings.parquet"
users = "Users.parquet"

# Load the datas for content-based recommendations
@st.cache_resource
def load_udemy_courses():
    # dataset = ParquetDataset(udemy_courses)
    df = pd.read_parquet(udemy_courses)
    tf_idf = TfidfVectorizer(stop_words="english")
    tf_idf_matrix = tf_idf.fit_transform(df["course_title"])
    cosine_sim = cosine_similarity(tf_idf_matrix)
    course_title_to_index = {title: i for i, title in enumerate(df["course_title"])}
    return df, cosine_sim, course_title_to_index

df, cosine_sim, course_title_to_index = load_udemy_courses()

# Load the datas for collaborative filtering recommendations
@st.cache_resource
def load_users_and_ratings():
    users_df = pd.read_parquet(users)
    ratings_df = pd.read_parquet(new_ratings)
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
    course_name = course_name.strip().lower()
    if course_name not in df['course_title'].str.lower().unique():
        return f"Course '{course_name}' not present in 'course_title' column in the dataset."

    course_id = df[df['course_title'].str.lower() == course_name]['course_id'].iloc[0]

    similar_courses = course_similarity_df[course_id]
    similar_courses = similar_courses.sort_values(ascending=False)
    recommended_course_ids = similar_courses.head(num_recommendations + 1).index.tolist()
    recommended_course_ids.remove(course_id) 
    recommended_courses = [df[df['course_id'] == cid]['course_title'].iloc[0] for cid in recommended_course_ids if cid in df['course_id'].values]
    return recommended_courses

def hybrid_recommender(course_name, content_weight=0.5):
    content_based_rec = content_based_recommender(course_name, df, cosine_sim, course_title_to_index)[:5]
    collaborative_filtering_rec = collaborative_recommender(course_name, user_course_matrix, course_similarity_df, df, num_recommendations=5)
    
    hybrid_rec = []
    for item in content_based_rec:
        hybrid_rec.append((item, content_weight))
    
    for item in collaborative_filtering_rec:
        hybrid_rec.append((item, 1 - content_weight))

    hybrid_rec.sort(key=lambda x: x[1], reverse=True)

    hybrid_rec = [item[0] for item in hybrid_rec]

    return hybrid_rec 

# Streamlit app
def recommend_courses(user_input, course, selected_tab_index):
    if user_input and course:
        if selected_tab_index is None:
            st.write("Click on any button to show different recommendations")
        else:
            if selected_tab_index == "Content-Based":
                recommendations = content_based_recommender(course, df, cosine_sim, course_title_to_index)
            elif selected_tab_index == "Collaborative":
                recommendations = collaborative_recommender(course, user_course_matrix, course_similarity_df, df, num_recommendations=5)
            elif selected_tab_index == "Hybrid":
                recommendations = hybrid_recommender(course)
            else:
                recommendations = None

            if isinstance(recommendations, str):
                st.write(recommendations)
            else:
                st.write("Here are some recommended courses:")
                for rec in recommendations:
                    st.write(f"- {rec}")
    else:
        st.write("Course not found.")

st.title("Udemy Course Recommender")

user_input = st.sidebar.text_input("Enter a subject or course title:", key="user_input")

# Create buttons for tabs
st.sidebar.markdown("## Choose Recommendation Type")
selected_tab_index = st.sidebar.radio("", ["Content-Based", "Collaborative", "Hybrid"], key="selected_tab_index")

if user_input:
    course = next((col for col in df['course_title'] if user_input in col), None)
    if not course:
        st.sidebar.warning("Course not found.")

# Display content based on selected tab
if user_input and course:
    recommend_courses(user_input, course, selected_tab_index)