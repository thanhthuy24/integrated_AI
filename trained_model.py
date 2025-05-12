# import pandas as pd
#
# # Đọc dữ liệu từ CSV
# df_views = pd.read_csv("dataMySQL/views.csv")  # Chứa user_id, course_id, view_count, completion_percentage
# df_courses = pd.read_csv("dataMySQL/course.csv")    # Chứa course_id, category_id
# df_categories = pd.read_csv("dataMySQL/category.csv")  # Chứa category_id, category_name
#
# # Đổi tên cột để chuẩn hóa
# df_categories.rename(columns={"id": "category_id", "name": "category_name"}, inplace=True)
#
# # Merge để có thông tin category của từng khóa học
# df_courses = df_courses.merge(df_categories, on="category_id", how="left")
# df_views = df_views.merge(df_courses, on="id", how="left")
#
# # print(df_views.head())
#
# # Tính tổng số lượt xem theo category cho mỗi user
# df_user_interest = df_views.groupby(["user_id", "category_name"])["view_count"].sum().reset_index()
#
# # Sắp xếp theo user_id và số lần xem (view_count)
# df_user_interest = df_user_interest.sort_values(by=["user_id", "view_count"], ascending=[True, False])
#
# print(df_user_interest.head())

# import pandas as pd
# import pickle
# from surprise import Dataset, Reader, SVD
# from surprise.model_selection import train_test_split
#
# # 📌 1️⃣ Đọc dữ liệu từ CSV
# df_views = pd.read_csv("dataMySQL/views.csv")
#
# # 📌 2️⃣ Chuẩn bị dữ liệu cho mô hình AI
# reader = Reader(rating_scale=(1, 20))  # Giả sử số lượt xem từ 1 - 20
# data = Dataset.load_from_df(df_views[['user_id', 'course_id', 'view_count']], reader)
#
# # 📌 3️⃣ Chia tập train & test
# trainset, testset = train_test_split(data, test_size=0.2)
#
# # 📌 4️⃣ Huấn luyện mô hình AI
# model = SVD()
# model.fit(trainset)
#
# # 📌 5️⃣ Lưu mô hình vào file "ai_model.pkl"
# with open("ai_model.pkl", "wb") as f:
#     pickle.dump(model, f)
#
# print("✅ Mô hình đã được train và lưu vào ai_model.pkl!")

import pandas as pd
import pickle
import psycopg2
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from datetime import datetime

# Kết nối PostgreSQL
def get_connection():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="view",
        user="postgres",
        password="Admin123@"
    )

# Đọc dữ liệu từ bảng views
def load_views_from_db():
    conn = get_connection()
    query = "SELECT user_id, course_id, view_count FROM view"
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def train_and_save_model():
    print(f"🚀 Bắt đầu huấn luyện model lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    df_views = load_views_from_db()

    # Chuẩn bị dữ liệu cho Surprise
    reader = Reader(rating_scale=(1, df_views["view_count"].max()))
    data = Dataset.load_from_df(df_views[['user_id', 'course_id', 'view_count']], reader)

    # Train/test split
    trainset, testset = train_test_split(data, test_size=0.2)

    # Train mô hình
    model = SVD()
    model.fit(trainset)

    # Lưu model
    with open("ai_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Đã huấn luyện và lưu model lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    train_and_save_model()
