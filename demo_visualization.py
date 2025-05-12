import matplotlib.pyplot as plt
import pandas as pd
import joblib

def load_data():
    try:
        model = joblib.load('xgb_model.pkl')
        recommendations = pd.read_csv('recommendations.csv')
        return model, recommendations
    except FileNotFoundError as e:
        print(f"Lỗi: Không tìm thấy file - {e}")
        print("Hãy chắc chắn bạn đã chạy file recommendation_system.py trước")
        exit(1)


def visualize_recommendations(user_id, recommendations):
    user_recs = recommendations[recommendations['user_id'] == user_id]

    if user_recs.empty:
        print(f"Không tìm thấy recommendations cho user {user_id}")
        return

    # Tạo visualization
    plt.figure(figsize=(10, 5))
    bars = plt.bar(user_recs['course_id'], user_recs['xgb_score'], color='skyblue')

    # Thêm giá trị số trên mỗi cột
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom')

    plt.title(f"Recommendations for {user_id}")
    plt.ylabel("Recommendation Score")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("🔄 Đang tải model và dữ liệu...")
    model, recommendations = load_data()

    # # Demo cho 3 user đầu tiên
    # sample_users = recommendations['user_id'].unique()[:5]
    #
    # for user in sample_users:
    #     print(f"\n📊 Đang hiển thị recommendations cho {user}...")
    #     visualize_recommendations(user, recommendations)
    try:
        user_id_input = input("Nhập user_id để xem recommendations: ")
        user_id = int(user_id_input)
        print(f"\n📊 Đang hiển thị recommendations cho user {user_id}...")
        visualize_recommendations(user_id, recommendations)
    except ValueError:
        print("❌ user_id không hợp lệ. Vui lòng nhập một số nguyên.")