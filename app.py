import os
import pickle
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from streamlit_option_menu import option_menu

# Suppress version mismatch warnings from scikit-learn models
warnings.filterwarnings('ignore')

try:
    from llm_helper import extract_text_from_file, parse_with_llm
    LLM_HELPER_AVAILABLE = True
except ImportError:
    LLM_HELPER_AVAILABLE = False

st.set_page_config(page_title="Multiple Model Prediction & Recommendation System", page_icon="🤖", layout="wide")

# Helper for robust model loading across different directory contexts
def load_saved_model(filename):
    paths = [
        filename,
        os.path.join('saved_models', filename),
        os.path.join(os.path.dirname(__file__), filename),
        os.path.join(os.path.dirname(__file__), 'saved_models', filename)
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p, 'rb') as f:
                return pickle.load(f)
    raise FileNotFoundError(f"Model file '{filename}' not found.")

# Loading the saved disease models
diabetes_model = load_saved_model('diabetes_model.sav')
heart_disease_model = load_saved_model('heart_disease_model.sav')
parkinsons_model = load_saved_model('parkinsons_model.sav')

# Loading course recommender model
@st.cache_resource
def load_course_recommender():
    candidates = ['course_recommendation.pkl', 'Saved_model.pkl']
    for filename in candidates:
        try:
            return load_saved_model(filename)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Course recommendation model file not found.")

course_artifacts = load_course_recommender()
cf_model = course_artifacts["cf_model"]
tfidf_matrix = course_artifacts["tfidf_matrix"]
course_index = course_artifacts["course_index"]
courses = course_artifacts["courses"]
alpha = course_artifacts["alpha"]
switch_threshold = course_artifacts["switch_threshold"]
min_rating = course_artifacts["min_rating"]

# Valid user ids the SVD model was actually trained on
valid_user_ids = {cf_model.trainset.to_raw_uid(i) for i in cf_model.trainset.all_users()}

# Helper functions for Course Recommendation
def dedupe_by_name(result_df, top_n):
    """Keep highest-scoring row per unique course_name."""
    return result_df.sort_values("score", ascending=False).drop_duplicates(subset="course_name").head(top_n)

def plot_scores(df):
    """Plot horizontal bar chart of recommended course scores."""
    if df.empty:
        st.warning("No recommendations to display.")
        return
    fig, ax = plt.subplots(figsize=(8, max(2, 0.4 * len(df))))
    ax.barh(df["course_name"][::-1], df["score"][::-1], color="#1f77b4")
    ax.set_xlabel("Recommendation / Similarity Score")
    st.pyplot(fig)
    plt.close(fig)

def normalize_scores(score_dict):
    vals = np.array(list(score_dict.values()))
    mn, mx = vals.min(), vals.max()
    if mx == mn:
        return {k: 0.5 for k in score_dict}
    return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}

def content_scores_for_course(course_id):
    idx = course_index[course_id]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    return pd.Series(sims, index=courses["course_id"].values)

def collab_scores_for_user(user_id):
    scores = {cid: cf_model.predict(user_id, cid).est for cid in courses["course_id"]}
    return pd.Series(scores)

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

    n_seed = len(seed_ids)
    eff_alpha = 0.7 if n_seed < switch_threshold else alpha
    hybrid = {cid: eff_alpha * cb_n.get(cid, 0) + (1 - eff_alpha) * cf_n.get(cid, 0)
              for cid in courses["course_id"]}
    return pd.Series(hybrid), seed_ids

# Sidebar for navigation
with st.sidebar:
    st.markdown("### Settings")
    api_key = st.text_input("Gemini API Key (for PDF/DOCX)", type="password")
    st.markdown("---")
    
    selected = option_menu('Multiple Model AI System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'Course Recommendation'],
                           menu_icon='cpu-fill',
                           icons=['activity', 'heart', 'person', 'journal-text'],
                           default_index=0)


# Diabetes Prediction Page
if selected == 'Diabetes Prediction':

    st.title('Diabetes Prediction using ML')

    st.markdown("### Upload Data for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV, PDF, or DOCX file", type=["csv", "pdf", "docx"], key="diabetes_file")
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            if not LLM_HELPER_AVAILABLE:
                st.error("llm_helper module is required for processing PDF/DOCX files.")
                st.stop()
            if not api_key:
                st.warning("Please enter your Gemini API Key in the sidebar to process PDF/DOCX files.")
                st.stop()
            with st.spinner("Extracting features using LLM..."):
                text = extract_text_from_file(uploaded_file)
                expected_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
                df = parse_with_llm(text, expected_features, api_key)
        st.write("Data Preview:", df.head())
        if st.button("Predict from CSV", key="diab_batch_btn"):
            try:
                X = df.drop(columns=['Outcome'], errors='ignore')
                predictions = diabetes_model.predict(X)
                df['Prediction'] = predictions
                df['Diagnosis'] = df['Prediction'].apply(lambda x: 'Diabetic' if x == 1 else 'Not Diabetic')
                st.write("Prediction Results:")
                st.dataframe(df, use_container_width=True)
                st.success("Batch Prediction Complete!")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
            
    st.markdown("### Or Enter Data Manually")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with col2:
        Glucose = st.text_input('Glucose Level')

    with col3:
        BloodPressure = st.text_input('Blood Pressure value')

    with col1:
        SkinThickness = st.text_input('Skin Thickness value')

    with col2:
        Insulin = st.text_input('Insulin Level')

    with col3:
        BMI = st.text_input('BMI value')

    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')

    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''

    if st.button('Diabetes Test Result', key="diab_btn"):
        user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                      BMI, DiabetesPedigreeFunction, Age]
        if any(x.strip() == '' for x in user_input):
            st.error("Please fill in all numerical input fields before submitting.")
        else:
            try:
                user_input = [float(x) for x in user_input]
                diab_prediction = diabetes_model.predict([user_input])

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'
            except ValueError:
                st.error("Please enter valid numeric values for all fields.")

    if diab_diagnosis:
        st.success(diab_diagnosis)


# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':

    st.title('Heart Disease Prediction using ML')

    st.markdown("### Upload Data for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV, PDF, or DOCX file", type=["csv", "pdf", "docx"], key="heart_file")
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            if not LLM_HELPER_AVAILABLE:
                st.error("llm_helper module is required for processing PDF/DOCX files.")
                st.stop()
            if not api_key:
                st.warning("Please enter your Gemini API Key in the sidebar to process PDF/DOCX files.")
                st.stop()
            with st.spinner("Extracting features using LLM..."):
                text = extract_text_from_file(uploaded_file)
                expected_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
                df = parse_with_llm(text, expected_features, api_key)
        st.write("Data Preview:", df.head())
        if st.button("Predict from CSV", key="heart_batch_btn"):
            try:
                X = df.drop(columns=['target'], errors='ignore')
                predictions = heart_disease_model.predict(X)
                df['Prediction'] = predictions
                df['Diagnosis'] = df['Prediction'].apply(lambda x: 'Heart Disease' if x == 1 else 'Healthy')
                st.write("Prediction Results:")
                st.dataframe(df, use_container_width=True)
                st.success("Batch Prediction Complete!")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
            
    st.markdown("### Or Enter Data Manually")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Age')

    with col2:
        sex = st.text_input('Sex')

    with col3:
        cp = st.text_input('Chest Pain types')

    with col1:
        trestbps = st.text_input('Resting Blood Pressure')

    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')

    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')

    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')

    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')

    with col3:
        exang = st.text_input('Exercise Induced Angina')

    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')

    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')

    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')

    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')

    heart_diagnosis = ''

    if st.button('Heart Disease Test Result', key="heart_btn"):
        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
        if any(x.strip() == '' for x in user_input):
            st.error("Please fill in all numerical input fields before submitting.")
        else:
            try:
                user_input = [float(x) for x in user_input]
                heart_prediction = heart_disease_model.predict([user_input])

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'
            except ValueError:
                st.error("Please enter valid numeric values for all fields.")

    if heart_diagnosis:
        st.success(heart_diagnosis)


# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":

    st.title("Parkinson's Disease Prediction using ML")

    st.markdown("### Upload Data for Batch Prediction")
    uploaded_file = st.file_uploader("Upload a CSV, PDF, or DOCX file", type=["csv", "pdf", "docx"], key="parkinsons_file")
    if uploaded_file is not None:
        filename = uploaded_file.name.lower()
        if filename.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            if not LLM_HELPER_AVAILABLE:
                st.error("llm_helper module is required for processing PDF/DOCX files.")
                st.stop()
            if not api_key:
                st.warning("Please enter your Gemini API Key in the sidebar to process PDF/DOCX files.")
                st.stop()
            with st.spinner("Extracting features using LLM..."):
                text = extract_text_from_file(uploaded_file)
                expected_features = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
                df = parse_with_llm(text, expected_features, api_key)
        st.write("Data Preview:", df.head())
        if st.button("Predict from CSV", key="park_batch_btn"):
            try:
                X = df.drop(columns=['name', 'status'], errors='ignore')
                predictions = parkinsons_model.predict(X)
                df['Prediction'] = predictions
                df['Diagnosis'] = df['Prediction'].apply(lambda x: 'Parkinsons' if x == 1 else 'Healthy')
                st.write("Prediction Results:")
                st.dataframe(df, use_container_width=True)
                st.success("Batch Prediction Complete!")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
            
    st.markdown("### Or Enter Data Manually")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with col2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with col3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with col1:
        RAP = st.text_input('MDVP:RAP')

    with col2:
        PPQ = st.text_input('MDVP:PPQ')

    with col3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with col1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with col2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with col3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with col1:
        HNR = st.text_input('HNR')

    with col2:
        RPDE = st.text_input('RPDE')

    with col3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with col1:
        D2 = st.text_input('D2')

    with col2:
        PPE = st.text_input('PPE')

    parkinsons_diagnosis = ''

    if st.button("Parkinson's Test Result", key="park_btn"):
        user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
        if any(x.strip() == '' for x in user_input):
            st.error("Please fill in all numerical input fields before submitting.")
        else:
            try:
                user_input = [float(x) for x in user_input]
                parkinsons_prediction = parkinsons_model.predict([user_input])

                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "The person has Parkinson's disease"
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson's disease"
            except ValueError:
                st.error("Please enter valid numeric values for all fields.")

    if parkinsons_diagnosis:
        st.success(parkinsons_diagnosis)


# Course Recommendation Page
if selected == 'Course Recommendation':

    st.title("📚 Course Recommendation System")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Content-Based (by Course)",
        "Collaborative (by User ID)",
        "Hybrid (by User ID)",
        "Dataset Info"
    ])

    # 1. Content-Based Recommendation
    with tab1:
        st.subheader("Content-Based Course Recommendation")
        selected_name = st.selectbox("Select a Course", sorted(courses["course_name"].unique()), key="cb_course")
        selected_id = courses[courses["course_name"] == selected_name]["course_id"].iloc[0]
        top_n = st.selectbox("Number of Recommendations", [5, 10, 15, 20], key="cb_top_n")

        if st.button("Recommend Courses", key="cb_btn"):
            scores = content_scores_for_course(selected_id)
            result = courses.copy()
            result["score"] = result["course_id"].map(scores)
            result = result[result["course_id"] != selected_id]
            result = dedupe_by_name(result, top_n)
            st.dataframe(result[["course_name", "difficulty_level", "course_price", "score"]],
                         use_container_width=True, hide_index=True)
            plot_scores(result)

    # 2. Collaborative Recommendation
    with tab2:
        st.subheader("Collaborative Course Recommendation")
        min_uid = int(min(valid_user_ids))
        max_uid = int(max(valid_user_ids))
        uid = st.number_input("Enter User ID", min_value=min_uid, max_value=max_uid, value=min_uid, step=1, key="cf_uid")
        top_n = st.selectbox("Number of Recommendations", [5, 10, 15, 20], key="cf_top_n")

        if st.button("Recommend Courses", key="cf_btn"):
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

    # 3. Hybrid Recommendation
    with tab3:
        st.subheader("Hybrid Course Recommendation")
        min_uid = int(min(valid_user_ids))
        max_uid = int(max(valid_user_ids))
        uid = st.number_input("Enter User ID", min_value=min_uid, max_value=max_uid, value=min_uid, step=1, key="hybrid_uid")
        top_n = st.selectbox("Number of Recommendations", [5, 10, 15, 20], key="hybrid_top_n")

        if st.button("Recommend Courses", key="hybrid_btn"):
            uid = int(uid)
            if uid not in valid_user_ids:
                st.error("User ID not found in the trained model.")
            else:
                scores, seed_ids = hybrid_scores_for_user(uid)
                result = courses.copy()
                result["score"] = result["course_id"].map(scores)
                result = result[~result["course_id"].isin(seed_ids)]
                result = dedupe_by_name(result, top_n)
                st.dataframe(result[["course_name", "difficulty_level", "course_price", "score"]],
                             use_container_width=True, hide_index=True)
                plot_scores(result)

    # 4. Dataset Information
    with tab4:
        st.subheader("Dataset & Model Information")

        st.markdown(f"**Total Trained Users:** {len(valid_user_ids):,}")
        st.markdown(f"**Total Courses Available:** {len(courses):,}")
        st.markdown("---")

        st.markdown("#### Content-Based Filtering")
        st.write("TF-IDF vectors built from course details (name, difficulty level, instructor, certification, study material, duration, price, feedback score) compared using cosine similarity.")

        st.markdown("#### Collaborative Filtering")
        st.write("SVD matrix factorization (Surprise library) trained on user-course ratings to predict ratings for user/course pairs.")

        st.markdown("#### Hybrid Filtering")
        st.write(f"Combines content and collaborative scores with weight `alpha = {alpha}` (adjusted dynamically based on available seed courses).")

        st.markdown("#### Course Catalog")
        st.dataframe(courses[["course_id", "course_name", "difficulty_level", "course_price", "feedback_score"]],
                     use_container_width=True, hide_index=True)
