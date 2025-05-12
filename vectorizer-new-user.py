import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import psycopg2


def get_connection_course():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="course",
        user="postgres",
        password="Admin123@"
    )


def get_lis_courses():
    conn = get_connection_course()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, tag_id, category_id, duration_hours
        FROM "course"
    """)
    rows = cursor.fetchall()
    conn.close()

    # Trả về list of dicts
    return [
        {
            "id": row[0],
            "name": row[1],
            "tag_id": row[2],
            "category_id": row[3],
            "duration_hours": row[4]
        }
        for row in rows
    ]


courses_list_df = pd.DataFrame(get_lis_courses())


# Tạo dữ liệu văn bản
def build_item_features(df):
    return [
        f"{row['id']} {row['name']} {row['tag_id']} {row['category_id']} {row['duration_hours']}"
        for _, row in df.iterrows()
    ]


item_features = build_item_features(courses_list_df)
print(item_features)

# Huấn luyện vectorizer
vectorizer = TfidfVectorizer()
item_features_matrix = vectorizer.fit_transform(item_features)

# Lưu mô hình và dữ liệu vector hóa
joblib.dump(vectorizer, 'new-user/tfidf_vectorizer.pkl')
joblib.dump(item_features_matrix, 'new-user/item_features_matrix.pkl')
courses_list_df.to_pickle('new-user/courses_df.pkl')

print("✅ TF-IDF và vector đặc trưng đã được lưu.")
