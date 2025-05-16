from flask import Flask, request, jsonify
# from test import fetch_dataframe
from textblob import TextBlob
import pickle
from transformers import pipeline
import psycopg2
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
# import trained_model
import joblib
import pandas as pd
from scipy import sparse
from features import extract_features
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_connection_course():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="course",
        user="postgres",
        password="Admin123@"
    )


def get_connection_enrollment():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="enrollment",
        user="postgres",
        password="Admin123@"
    )

def get_lis_courses():
    conn = get_connection_course()
    cursor = conn.cursor()
    cursor.execute("""
            SELECT *
            FROM "course"
        """)
    result = [row[0] for row in cursor.fetchall()]
    conn.close()
    return result

# Load mô hình đã train ----
# model_path_1 = "ai_model.pkl"
# with open(model_path_1, "rb") as f:
#     model = pickle.load(f)

import test
# Hàm cập nhật model + log thời gian
def update_model_and_log():
    try:
        test.train_and_save_model()
        print(f"🔁 Model đã được cập nhật lại lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print(f"❌ Lỗi khi cập nhật model: {e}")


# Khởi tạo và lên lịch cho APScheduler
scheduler = BackgroundScheduler()
scheduler.add_job(update_model_and_log, 'interval', minutes=5)
scheduler.start()

app = Flask(__name__)


def fetch_dataframe(connection_func, query):
    conn = connection_func()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Load mô hình và encoder
model_als = joblib.load('model_als.pkl')
model_xgb = joblib.load('model_xgb_ranker.pkl')
user_encoder = joblib.load('user_encoder.pkl')
course_encoder = joblib.load('course_encoder.pkl')

user_item_matrix = sparse.load_npz("user_items.npz")


def generate_dummy_data():
    interactions = fetch_dataframe(get_connection_enrollment, "SELECT * FROM interactions")
    wishlist = fetch_dataframe(get_connection_course, "SELECT * FROM wishlist")
    ratings = fetch_dataframe(get_connection_enrollment, "SELECT * FROM courserating")
    course_metadata = fetch_dataframe(get_connection_course, "SELECT * FROM course")

    global_avg_rating = ratings['rating'].mean()

    return interactions, wishlist, ratings, course_metadata, global_avg_rating


interactions, wishlist, ratings, course_metadata, global_avg_rating = generate_dummy_data()


@app.route("/recommend", methods=["GET"])
def recommend():
    try:
        user_id = request.args.get("user_id", type=int)
        if user_id is None:
            return {"error": "Thiếu user_id trong query"}

        if user_id not in user_encoder.classes_:
            return {"error": f"User {user_id} chưa có trong dữ liệu training."}

        encoded_user = user_encoder.transform([user_id])[0]

        if encoded_user >= user_item_matrix.shape[0]:
            return {"error": f"user_item_matrix không chứa hàng cho user {user_id} (index {encoded_user})"}

        user_vector = user_item_matrix[encoded_user]

        # Gợi ý từ mô hình ALS
        indices, scores = model_als.recommend(
            userid=encoded_user,
            user_items=user_vector,
            N=50,
            filter_already_liked_items=True
        )

        recommended_course_idxs = [idx for idx, _ in zip(indices, scores)]
        recommended_course_ids = course_encoder.inverse_transform(recommended_course_idxs)

        # Tính toán đặc trưng đầy đủ cho từng (user_id, course_id)
        feature_rows = []
        for course_id in recommended_course_ids:
            try:
                features = extract_features(
                    user_id=user_id,
                    course_id=course_id,
                    interactions=interactions,
                    wishlist=wishlist,
                    ratings=ratings,
                    course_metadata=course_metadata,
                    global_avg_rating=global_avg_rating
                )
                feature_rows.append(features)
            except Exception as fe:
                print(f"Bỏ qua course_id {course_id} do lỗi khi trích xuất đặc trưng: {fe}")

        if not feature_rows:
            return {"error": "Không có khóa học nào hợp lệ để đề xuất."}

        candidate_df = pd.DataFrame(feature_rows)

        candidate_df = pd.get_dummies(candidate_df, columns=["category", "difficulty"])

        # Thêm tính toán raw_score có thể giải thích được
        candidate_df['raw_score'] = (
                candidate_df['is_purchased'] * 1.5 +
                candidate_df['in_wishlist'] * 3 +
                candidate_df['total_clicks'] * 0.5
        )

        expected_features = model_xgb.get_booster().feature_names

        for feature in expected_features:
            if feature not in candidate_df.columns:
                candidate_df[feature] = 0

        X_features = candidate_df[expected_features]
        candidate_df['xgb_score'] = model_xgb.predict(X_features)

        # Sắp xếp và lấy top 5
        top = (
            candidate_df
            .sort_values('xgb_score', ascending=False)
            .head(8)
            [['user_id', 'course_id', 'xgb_score']]
        )

        return top["course_id"].tolist()
        # return top.to_dict(orient='records')

    except Exception as e:
        return {"error": str(e)}


vectorizer = joblib.load('new-user/tfidf_vectorizer.pkl')
item_features_matrix = joblib.load('new-user/item_features_matrix.pkl')
courses_df = pd.read_pickle('new-user/courses_df.pkl')


def recommend_for_new_user(profile, top_k=8):
    # category_text = ' '.join(profile['category_id']) if isinstance(profile['category_id'], list) else profile[
    #     'category_id']

    if isinstance(profile['category_id'], list):
        category_text = ' '.join(str(c) for c in profile['category_id'])
    else:
        category_text = str(profile['category_id'])

    profile_text = f"{profile['name']} {profile['tag_id']} {category_text} {profile['duration_hours']}"
    user_vector = vectorizer.transform([profile_text])
    similarities = cosine_similarity(user_vector, item_features_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    return courses_df.iloc[top_indices].to_dict(orient='records')


# import gen_data
@app.route('/recommend-new-user', methods=['POST'])
def recommend_new_user():
    data = request.get_json()

    print("Received data:", data)

    required_fields = ['name', 'tag_id', 'category_id', 'duration_hours']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields'}), 400

    recommendations = recommend_for_new_user(data)
    course_ids = [course['id'] for course in recommendations]

    # result = gen_data.evaluate_recommendations(data, courses_df, k=10)
    # pd.DataFrame([result]).to_csv("data_update/evaluation_metrics_new_user.csv", index=False)

    return jsonify(course_ids)
    # return jsonify({'recommended_courses': recommendations})


# @app.route('/rcm', methods=['GET'])
# def rcm():
#     user_id = request.args.get("user_id", type=int)
#     if user_id is None:
#         return jsonify({"error": "user_id is required"}), 400
#
#     try:
#         course_ids = get_all_course_ids()
#         train_user_ids = set(model.trainset._raw2inner_id_users.keys())
#
#         if user_id not in train_user_ids:
#             recommended_courses = get_popular_courses()
#         else:
#             predictions = [(cid, model.predict(user_id, cid).est) for cid in course_ids]
#             predictions.sort(key=lambda x: x[1], reverse=True)
#             recommended_courses = [x[0] for x in predictions[:5]]
#
#         return jsonify(recommended_courses)
#
#     except Exception as e:
#         print("❌ Exception occurred:", e)
#         return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    # print("Received data:", data)  # Debug
    content = data.get("comment", "")


    if not content:
        return jsonify({"error": "No comment provided"}), 400

    # Phân tích cảm xúc bằng TextBlob
    polarity = TextBlob(content).sentiment.polarity

    # Trả về kết quả dưới dạng JSON
    return jsonify({"sentiment_score": polarity})


@app.route('/analyze-rating', methods=['POST'])
def analyze_sentiment_rating():
    data = request.get_json()
    # print("Received data:", data)  # Debug
    content = data.get("comment", "")

    if not content:
        return jsonify({"error": "No comment provided"}), 400

    # Phân tích cảm xúc bằng TextBlob
    polarity = TextBlob(content).sentiment.polarity

    # Trả về kết quả dưới dạng JSON
    return jsonify({"sentiment_score": polarity})


# model_path = "E:/CNPM/integrated-AI/distilbert_model"
# classifier = pipeline("text-classification", model=model_path)

model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
classifier = pipeline("text-classification", model=model_name, tokenizer=model_name)

label_map = {
    "LABEL_0": "ham",
    "LABEL_1": "spam"
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    comment = data.get("comment", "")

    if not comment:
        return jsonify({"error": "No comment provided"}), 400

    result = classifier(comment)[0]
    label = label_map.get(result["label"], "unknown")

    return jsonify({
        "label": label,
        "score": result["score"]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
