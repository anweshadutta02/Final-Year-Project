import streamlit as st
from streamlit.components.v1 import html
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#Styles
button_style = """
<style>
  .stButton > button {
    background-color: #F0F2F5; /* Light gray background */
    color: #343A40; /* Dark gray text */
    border: none;
    border-radius: 4px;
    padding: 10px 20px;
    font-size: 16px;
    cursor: pointer;
  }
  .stButton > button:hover {
    background-color: #E0E4E8; /* Slightly darker on hover */
  }
</style>"""

# Load the datas for content-based recommendations
df = pd.read_csv("udemy_courses.csv")

# Create TF-IDF matrix for content-based recommendations
tf_idf = TfidfVectorizer(stop_words="english")
tf_idf_matrix = tf_idf.fit_transform(df["course_title"])
cosine_sim = cosine_similarity(tf_idf_matrix)

# Create a dictionary for faster title lookup for content-based recommendations
course_title_to_index = {title: i for i, title in enumerate(df["course_title"])}

# Load the datas for collaborative filtering recommendations
users_df = pd.read_csv('Users.csv')
ratings_df = pd.read_csv('new_ratings.csv')
df2 = pd.merge(ratings_df, users_df, on='User-ID')
df2 = pd.merge(df2, df, on='course_id')
user_course_matrix = df2.pivot_table(index='User-ID', columns='course_id', values='Rating')
user_course_matrix = user_course_matrix.fillna(0)

# Create cosine similarity matrix for collaborative filtering recommendations
course_similarity = cosine_similarity(user_course_matrix.T)
course_similarity_df = pd.DataFrame(course_similarity, index=user_course_matrix.columns, columns=user_course_matrix.columns)


def content_based_recommender(title):
    """
    Recommends courses similar to the given title using content-based filtering.

    Args:
        title (str): The title of the course to base recommendations on.

    Returns:
        list: A list of recommended course titles.
    """

    course_index = course_title_to_index.get(title)
    if course_index is None:
        return ["Course not found."]

    similarity_scores = pd.DataFrame(cosine_sim[course_index], columns=["score"])
    course_indices = similarity_scores.sort_values(
        "score", ascending=False
    )[1:5].index

    return df["course_title"].iloc[course_indices]


def recommend_courses(course_name, num_recommendations=5):
    # Clean and preprocess course name (remove leading/trailing whitespaces, make it lowercase)
    course_name = course_name.strip().lower()

    # Check if the course name exists in the 'course_title' column of the original DataFrame 'df'
    if course_name not in df['course_title'].str.lower().unique():
        return f"Course '{course_name}' not present in 'course_title' column in the dataset."

    # Find the corresponding course ID from 'course_id' column
    course_id = df[df['course_title'].str.lower() == course_name]['course_id'].iloc[0]

    similar_courses = course_similarity_df[course_id]
    similar_courses = similar_courses.sort_values(ascending=False)
    recommended_courses = similar_courses.head(num_recommendations + 1).index.tolist()
    recommended_courses.remove(course_id)  # Remove the input course itself
    return recommended_courses


def collaborative_recommender(course):
    """
    Recommends courses similar to the given title using collaborative filtering
    (placeholder, not implemented).

    Args:
        title (str): The title of the course to base recommendations on.

    Returns:
        list: A list of recommended course titles (currently placeholder).
    """

    # recommendations = recommend_courses(course, num_recommendations=10)

    # if isinstance(recommendations, str):
    #     return [recommendations]  # Return error message as a list

    # recommended_titles = [df[df['course_id'] == course]['course_title'].values[0] for course in recommendations]
    # return recommended_titles
    return ["None as of yet."]


def hybrid_recommender(title):
    """
    Recommends courses similar to the given title using a hybrid approach
    (placeholder, not implemented).

    Args:
        title (str): The title of the course to base recommendations on.

    Returns:
        list: A list of recommended course titles (currently placeholder).
    """

    return ["Hybrid recommendations are not yet implemented."]


# Streamlit app
st.title("Udemy Course Recommender")
user_input = st.text_input("Enter a subject or course title:")
course = [col for col in df['course_title'] if user_input in col][0]

if "selected_tab_index" not in st.session_state:
    st.session_state["selected_tab_index"] = 0

# Create a column for horizontal button layout
col1, col2, col3 = st.columns([1, 1, 1])

# Create buttons for tabs
content_based_button = col1.button("Content-Based")
collaborative_button = col2.button("Collaborative")
hybrid_button = col3.button("Hybrid")
st.markdown(button_style, unsafe_allow_html=True)

# Update selected tab index based on button clicks
if content_based_button:
    st.session_state["selected_tab_index"] = 0
elif collaborative_button:
    st.session_state["selected_tab_index"] = 1
elif hybrid_button:
    st.session_state["selected_tab_index"] = 2

# Get the currently selected tab index
selected_tab_index = st.session_state["selected_tab_index"]

# Display content based on selected tab
if user_input:
    if selected_tab_index == 0:
        recommendations = content_based_recommender(course)
    elif selected_tab_index == 1:
        recommendations = collaborative_recommender(course)
    elif selected_tab_index == 2:
        recommendations = hybrid_recommender(user_input)
    else:
        recommendations = []

    if len(recommendations) == 1:
        st.write(recommendations[0])
    else:
        st.write("Here are some recommended courses:")
        for course in recommendations:
            st.write(f"- {course}")
