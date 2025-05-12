import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import implicit
from xgboost import XGBRanker
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
from scipy import sparse
import features
import pandas as pd
import psycopg2

# --------------------------
# 1. TẠO DỮ LIỆU GIẢ LẬP
# --------------------------


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


def fetch_dataframe(connection_func, query):
    conn = connection_func()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def generate_dummy_data():
    interactions = pd.read_csv('data_update/interactions.csv', quotechar='"')
    wishlist = pd.read_csv('data_update/wishlist.csv', quotechar='"')
    ratings = pd.read_csv('data_update/courserating.csv', quotechar='"')
    course_metadata = pd.read_csv('data_csv/course.csv')

    return interactions, wishlist, ratings, course_metadata


interactions, wishlist, ratings, course_metadata = generate_dummy_data()

# --------------------------
# 2. XỬ LÝ DỮ LIỆU CHO IMPLICIT ALS
# --------------------------
# Fix OpenBLAS warning
os.environ['OPENBLAS_NUM_THREADS'] = '1'

user_encoder = LabelEncoder()
course_encoder = LabelEncoder()

interactions['user_encoded'] = user_encoder.fit_transform(interactions['user_id'])
interactions['course_encoded'] = course_encoder.fit_transform(interactions['course_id'])

# Tính trọng số
interactions['weight'] = interactions['purchased'] * 3 + interactions['clicks'] * 0.2

# Chuyển sang CSR format (Implicit yêu cầu)
user_item_matrix = csr_matrix(
    (interactions['weight'],
     (interactions['user_encoded'], interactions['course_encoded'])),
     shape=(len(user_encoder.classes_), len(course_encoder.classes_))
)

# --------------------------
# 3. HUẤN LUYỆN IMPLICIT ALS
# --------------------------
print("\n🔄 Training Implicit ALS model...")
model_implicit = implicit.als.AlternatingLeastSquares(
    factors=64,
    iterations=20,
    calculate_training_loss=True,
    random_state=42
)

model_implicit.fit(user_item_matrix)

# Đề xuất candidates
user_id_map = {user: idx for idx, user in enumerate(user_encoder.classes_)}
candidates = {}
for user in interactions['user_id'].unique():
    user_encoded = user_id_map[user]
    recommended_items, scores = model_implicit.recommend(
        userid=user_encoded,
        user_items=user_item_matrix[user_encoded],
        N=50,
        filter_already_liked_items=True
    )
    candidates[user] = [course_encoder.inverse_transform([course_id])[0] for course_id in recommended_items]

print("✅ Generated candidates for all users")

# --------------------------
# 4. CHUẨN BỊ DỮ LIỆU CHO XGBOOST
# --------------------------
# Tính rating trung bình toàn hệ thống
global_avg_rating = ratings['rating'].mean()

features_list = [
    features.extract_features(user, course, interactions, wishlist, ratings, course_metadata, global_avg_rating)
    for user, courses in candidates.items()
    for course in courses
]

features_df = pd.DataFrame(features_list)
features_df = pd.get_dummies(features_df, columns=['category', 'difficulty'])

# Tạo label với trọng số tùy chỉnh (ưu tiên mua > rating > wishlist > click)
features_df['raw_score'] = (
    features_df['is_purchased'] * 1.5 +
    features_df['in_wishlist'] * 3 +
    features_df['total_clicks'] * 0.5
)

# Scale về khoảng [1, 10] và ép kiểu int (cần thiết cho XGBRanker)
scaler = MinMaxScaler(feature_range=(1, 10))
features_df['label'] = scaler.fit_transform(features_df[['raw_score']]).astype(int)

# Xóa cột tạm nếu không cần
# features_df.drop(columns=['raw_score'], inplace=True)

# --------------------------
# 5. HUẤN LUYỆN XGBOOST RANKER
# --------------------------
print("\n🔄 Training XGBoost Ranker...")

# Reset index để đảm bảo phù hợp
features_df = features_df.reset_index(drop=True)

# Chia dữ liệu theo user
unique_users = features_df['user_id'].unique()
train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)

train_mask = features_df['user_id'].isin(train_users)
test_mask = features_df['user_id'].isin(test_users)

X_train = features_df[train_mask].drop(['user_id', 'course_id', 'label'], axis=1)
y_train = features_df[train_mask]['label']
X_test = features_df[test_mask].drop(['user_id', 'course_id', 'label'], axis=1)
y_test = features_df[test_mask]['label']

# Tính toán group sizes
train_groups = features_df[train_mask].groupby('user_id').size().values
test_groups = features_df[test_mask].groupby('user_id').size().values

model_xgb = XGBRanker(
    objective='rank:pairwise',
    tree_method='hist',
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)

model_xgb.fit(
    X_train,
    y_train,
    group=train_groups,
    verbose=True
)

print("✅ XGBoost training completed")

# --------------------------
# 6. DỰ ĐOÁN VÀ HIỂN THỊ KẾT QUẢ
# --------------------------
features_df['xgb_score'] = model_xgb.predict(features_df.drop(['user_id', 'course_id', 'label'], axis=1))
top_recommendations = (
    features_df
    .sort_values(['user_id', 'xgb_score'], ascending=[True, False])
    .groupby('user_id')
    .head(8)
)

# Lưu model và dữ liệu cần thiết
import joblib

# Sau khi huấn luyện xong
joblib.dump(model_xgb, 'xgb_model.pkl')

# Lưu các mô hình và encoder
joblib.dump(model_implicit, 'model_als.pkl')
joblib.dump(model_xgb, 'model_xgb_ranker.pkl')
joblib.dump(user_encoder, 'user_encoder.pkl')
joblib.dump(scaler, "scaler.pkl")
joblib.dump(course_encoder, 'course_encoder.pkl')

top_recommendations.to_csv('recommendations.csv', index=False)

# user_item_matrix là ma trận sparse được dùng trong model.implicit.recommend
sparse.save_npz("user_items.npz", user_item_matrix)
print("✅ Saved user_items.npz thành công")

# import gen_data
# result = gen_data.evaluate_recommendations(top_recommendations, features_df, k=10)
# pd.DataFrame([result]).to_csv("data_update/evaluation_metrics.csv", index=False)


# Giả sử bạn có ground truth từ lịch sử tương tác:
# ground_truth = features_df[features_df['raw_score'] > 0][['user_id', 'course_id']].drop_duplicates()
#
# import gen_data
#
# # Đánh giá top 5 đề xuất
# result = gen_data.evaluate_recommendations(top_recommendations, ground_truth, k=10)
#
# print("\n🎯 Evaluation Metrics:")
# for metric, value in result.items():
#     print(f"{metric}: {value:.4f}")
#

