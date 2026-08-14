# Importing the necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Course Recommender", layout="wide")

# Loading the Model pkl file
@st.cache_resource
def saved_model():
    with open("Saved_model.pkl", "rb") as f:
        return pickle.load(f)

# Getting meta datas from saved model
artifacts        = saved_model()
cf_model         = artifacts["cf_model"]
tfidf_matrix     = artifacts["tfidf_matrix"]
course_index     = artifacts["course_index"]
courses          = artifacts["courses"]
alpha            = artifacts["alpha"]
switch_threshold = artifacts["switch_threshold"]
min_rating       = artifacts["min_rating"]

# Valid user ids the SVD model was actually trained on
valid_user_ids = {cf_model.trainset.to_raw_uid(i) for i in cf_model.trainset.all_users()}

# Removing the duplicate coursenames from predictions
def dedupe_by_name(result_df, top_n):
    """
    Keep highest-scoring row per unique course_name.
    """
    return result_df.sort_values("score", ascending=False).drop_duplicates(subset="course_name").head(top_n)

# After getting the course output informations plotting bars the priority of course
def plot_scores(df):
    fig, ax = plt.subplots(figsize=(6, 0.4 * len(df) + 1))
    ax.barh(df["course_name"][::-1], df["score"][::-1])
    ax.set_xlabel("Similarity Score")
    st.pyplot(fig)

def normalize_scores(score_dict):
    vals = np.array(list(score_dict.values()))
    mn, mx = vals.min(), vals.max()
    if mx == mn:
        return {k: 0.5 for k in score_dict}
    return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}

# Getting the content based scores (course vs every other course, via TF-IDF)
def content_scores_for_course(course_id):
    idx = course_index[course_id]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    return pd.Series(sims, index=courses["course_id"].values)

# Getting the collaborative filtering scores (SVD prediction per course, for one user)
def collab_scores_for_user(user_id):
    scores = {cid: cf_model.predict(user_id, cid).est for cid in courses["course_id"]}
    return pd.Series(scores)

# Hybrid scores from both content and collaborative
# NOTE: this pkl does not include the original ratings history (df), so unlike the
# notebook's hybrid_recommend, we cannot look up a user's actually-liked courses.
# As a stand-in, the courses the SVD model predicts highest for that user are used
# as the "liked" seed set to build a content profile, then blended with the CF scores.
def hybrid_scores_for_user(user_id, seed_n=5):
    cf_scores = collab_scores_for_user(user_id)

    seed_ids = cf_scores.sort_values(ascending=False).head(seed_n).index
    seed_ids = [c for c in seed_ids if c in course_index]

    cb_raw = np.zeros(tfidf_matrix.shape[0])
    for cid in seed_ids:
        cb_raw += cosine_similarity(tfidf_matrix[course_index[cid]], tfidf_matrix).flatten()
    if seed_ids:
        cb_raw /= len(seed_ids)
    cb_scores = pd.Series(cb_raw, index=courses["course_id"].values)

    cf_n = normalize_scores(cf_scores.to_dict())
    cb_n = normalize_scores(cb_scores.to_dict())

    n_seed  = len(seed_ids)
    eff_alpha = 0.7 if n_seed < switch_threshold else alpha
    hybrid = {cid: eff_alpha * cb_n.get(cid, 0) + (1 - eff_alpha) * cf_n.get(cid, 0)
              for cid in courses["course_id"]}
    return pd.Series(hybrid), seed_ids

# Sidebar Navigations
page = st.sidebar.radio(
    "Navigate",
    ["Dataset Info", "Content-Based (by Course)", "Collaborative (by User ID)", "Hybrid (by User ID)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"Total Users (trained): {len(valid_user_ids):,}")
st.sidebar.markdown(f"Total Courses: {len(courses):,}")

# Dataset Informations
# shows the all data informations
if page == "Dataset Info":
    st.title("Dataset Information")

    st.subheader("Content-Based Filtering")
    st.write("TF-IDF vectors built from course name, difficulty level, instructor, "
             "certification, study material, duration, price and feedback score, "
             "compared using cosine similarity.")

    st.subheader("Collaborative Filtering")
    st.write("SVD matrix factorization (Surprise library) trained on user-course ratings. "
             "Predicts a rating for any user/course pair with `cf_model.predict()`.")

    st.subheader("Hybrid")
    st.write(f"hybrid_score = alpha x CB_norm + (1 - alpha) x CF_norm, alpha = {alpha} "
             f"(boosted to 0.7 when fewer than {switch_threshold} seed courses are available). "
             "Since this saved model doesn't include ratings history, the user's top "
             "SVD-predicted courses are used as a stand-in for their liked courses.")

    st.subheader("Course Catalog")
    st.dataframe(courses[["course_id", "course_name", "difficulty_level", "course_price",
                          "feedback_score"]], use_container_width=True, hide_index=True)

# Content Based recommendation
elif page == "Content-Based (by Course)":
    st.title("Content-Based Recommendation")
    # selecting the course names and max number of recommendation
    selected_name = st.selectbox("Select a Course", sorted(courses["course_name"].unique()))
    selected_id = courses[courses["course_name"] == selected_name]["course_id"].iloc[0]
    top_n = st.selectbox("Number of Recommendations", [5, 10, 15, 20])
    # click to findout the recommendations
    if st.button("Recommend"):
        scores = content_scores_for_course(selected_id)
        result = courses.copy()
        result["score"] = result["course_id"].map(scores)
        result = result[result["course_id"] != selected_id]
        result = dedupe_by_name(result, top_n)
        st.dataframe(result[["course_name", "difficulty_level", "course_price", "score"]],
                     use_container_width=True, hide_index=True)
        plot_scores(result)

# Collaborative Filtering recommendation
elif page == "Collaborative (by User ID)":
    st.title("Collaborative Recommendation")
    # selecting the user id and max number of recommendation
    uid = st.number_input("Enter User ID", min_value=int(min(valid_user_ids)),
                           max_value=int(max(valid_user_ids)), value=int(min(valid_user_ids)), step=1)
    top_n = st.selectbox("Number of Recommendations", [5, 10, 15, 20])

    if st.button("Recommend"):
        uid = int(uid)
        if uid not in valid_user_ids:
            st.error("User ID not found in the trained model.")
        else:
            scores = collab_scores_for_user(uid)
            result = courses.copy()
            result["score"] = result["course_id"].map(scores)
            result = dedupe_by_name(result, top_n)
            st.dataframe(result[["course_name", "difficulty_level", "course_price", "score"]],
                         use_container_width=True, hide_index=True)
            plot_scores(result)

# Hybrid
elif page == "Hybrid (by User ID)":
    st.title("Hybrid Recommendation")

    uid = st.number_input("Enter User ID", min_value=int(min(valid_user_ids)),
                           max_value=int(max(valid_user_ids)), value=int(min(valid_user_ids)), step=1)
    top_n = st.selectbox("Number of Recommendations", [5, 10, 15, 20])
    # this gets both of the content scores and collaborative
    if st.button("Recommend"):
        uid = int(uid)
        if uid not in valid_user_ids:
            st.error("User ID not found in the trained model.")
        else:
            scores, seed_ids = hybrid_scores_for_user(uid)
            result = courses.copy()
            result["score"] = result["course_id"].map(scores)
            result = result[~result["course_id"].isin(seed_ids)]
            # Taking the hybrid result removing duplicates
            result = dedupe_by_name(result, top_n)
            st.dataframe(result[["course_name", "difficulty_level", "course_price", "score"]],
                         use_container_width=True, hide_index=True)
            plot_scores(result)
