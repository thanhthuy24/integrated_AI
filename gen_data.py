# import csv
# import random
#
# filename = 'data_csv/course.csv'
# rows = []
#
# # Đọc và cập nhật dữ liệu
# with open(filename, mode='r', encoding='utf-8') as file:
#     reader = csv.DictReader(file)
#     fieldnames = reader.fieldnames
#     for row in reader:
#         if row['duration_hours'].lower() == 'null': # Nếu cột này trống
#             row['duration_hours'] = str(random.randint(1, 20))
#         rows.append(row)
#
# # Ghi lại vào file (có thể ghi đè hoặc tạo file mới nếu muốn giữ bản gốc)
# with open(filename, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.DictWriter(file, fieldnames=fieldnames)
#     writer.writeheader()
#     writer.writerows(rows)
#
# print(f"Đã thêm dữ liệu vào cột 'duration_hours' trong '{filename}'.")

# ----- Tạo file tương tác giữa user và course -------

# import pandas as pd
#
# # Đọc các file CSV
# views_df = pd.read_csv('data_csv/view.csv')
# enrollments_df = pd.read_csv('data_csv/enrollment.csv')
#
# # Gộp số lượt xem (clicks) theo user_id và course_id
# clicks_df = views_df.groupby(['user_id', 'course_id'])['view_count'].sum().reset_index()
# clicks_df.rename(columns={'view_count': 'clicks'}, inplace=True)
#
# # Đánh dấu các lượt mua (purchased)
# enrollments_df['purchased'] = 1
# purchased_df = enrollments_df[['user_id', 'course_id', 'purchased']]
#
# # Gộp dữ liệu từ cả hai nguồn
# interactions = pd.merge(clicks_df, purchased_df, on=['user_id', 'course_id'], how='outer')
#
# # Nếu user xem nhưng không mua, thì purchased = 0
# interactions['clicks'] = interactions['clicks'].fillna(0).astype(int)
# interactions['purchased'] = interactions['purchased'].fillna(0).astype(int)
#
# interactions.insert(0, 'id', range(5, 5 + len(interactions)))
#
# # Xuất ra file CSV
# interactions.to_csv('interactions.csv', index=False)
#
# print("✅ Đã tạo file 'interactions.csv' thành công!")

# -------------- Tạo dữ liệu giả cho file wishlist -------------------

# import pandas as pd
# import random
#
# # Đọc file user và course
# users_df = pd.read_csv('data_csv/user.csv')
# courses_df = pd.read_csv('data_csv/course.csv')
#
# user_ids = users_df['id'].unique()
# course_ids = courses_df['id'].unique()
#
# # Tạo danh sách wishlist giả
# wishlist_entries = []
# id_counter = 1
#
# # Giả sử mỗi user sẽ thích từ 1 đến 5 khóa học
# for user_id in user_ids:
#     num_courses = random.randint(1, 5)
#     liked_courses = random.sample(list(course_ids), min(num_courses, len(course_ids)))
#     for course_id in liked_courses:
#         wishlist_entries.append({'id': id_counter, 'user_id': user_id, 'course_id': course_id})
#         id_counter += 1
#
# # Tạo DataFrame và lưu ra file CSV
# wishlist_df = pd.DataFrame(wishlist_entries)
# wishlist_df.to_csv('wishlist.csv', index=False)
#
# print("✅ Đã tạo file 'wishlist.csv' với dữ liệu giả.")

# ---------------- tạo dữ liệu giả cho file interesting.csv----------------------------

# import pandas as pd
# import random
#
# # Đọc dữ liệu user và category
# users_df = pd.read_csv('data_csv/user.csv')
# categories_df = pd.read_csv('data_csv/category.csv')
#
# user_ids = users_df['id'].tolist()
# category_ids = categories_df['id'].tolist()
#
# data = []
# current_id = 8
#
# # Gán từ 3 đến 6 category yêu thích cho mỗi user
# for user_id in user_ids:
#     num_interests = random.randint(3, 6)
#     selected_categories = random.sample(category_ids, num_interests)
#     for category_id in selected_categories:
#         data.append({'id': current_id, 'user_id': user_id, 'category_id': category_id})
#         current_id += 1
#
# # Tạo DataFrame và ghi ra file
# interests_df = pd.DataFrame(data)
# interests_df.to_csv('user_interest.csv', index=False)
#
# print("✅ Đã tạo file 'user_interest.csv' thành công!")

# ---------------- tạo dữ liệu giả cho file category_job ----------------------------

# import pandas as pd
#
# # Define the mapping from category to related jobs in English
# data = [
#     ("TOEIC", ["Translator", "Interpreter", "English Teacher", "Import/Export Staff"]),
#     ("Web development", ["Web Developer", "Frontend Developer", "Backend Developer", "Full Stack Developer"]),
#     ("AI", ["AI Engineer", "Machine Learning Engineer", "Data Scientist", "Research Scientist"]),
#     ("Develop yourself", ["Personal Development Coach", "HR Trainer", "Mentor", "Soft Skills Trainer"]),
#     ("Graphic design", ["Graphic Designer", "UI Designer", "Visual Content Creator", "Branding Specialist"]),
#     ("Office Informatics", ["Office Staff", "Administrative Assistant", "Accountant", "Data Entry Specialist"]),
#     ("Programming", ["Software Developer", "Mobile App Developer", "Game Developer", "Embedded System Engineer"]),
#     ("Data Science", ["Data Analyst", "Data Scientist", "BI Analyst", "Data Engineer"]),
#     ("Business", ["Business Analyst", "Business Development Executive", "Sales Manager", "Operations Manager"]),
#     ("Design", ["Product Designer", "UX Designer", "Creative Director", "Industrial Designer"]),
#     ("Marketing", ["Digital Marketer", "SEO Specialist", "Content Marketer", "Marketing Manager"]),
#     ("Finance", ["Financial Analyst", "Accountant", "Investment Banker", "Auditor"]),
#     ("Health & Fitness", ["Personal Trainer", "Nutrition Specialist", "Fitness Coach", "Health Coach"]),
#     ("Music", ["Musician", "Music Teacher", "Sound Engineer", "Producer"]),
#     ("Photography", ["Photographer", "Photo Editor", "Visual Storyteller", "Studio Assistant"]),
#     ("Personal Development", ["Life Coach", "Career Counselor", "Motivational Speaker", "HR Trainer"])
# ]
#
# # Create a DataFrame
# df = pd.DataFrame(data, columns=["Category", "Related Jobs"])
#
# # Convert list of jobs to semicolon-separated string
# df["Related Jobs"] = df["Related Jobs"].apply(lambda jobs: "; ".join(jobs))
#
# # Save to CSV
# file_path = "/mnt/data/category_to_jobs.csv"
# df.to_csv(file_path, index=False)
#
# file_path

# ---------------- tạo dữ liệu giả cho file overview.csv----------------------------

# import pandas as pd
# import numpy as np
# import random
#
# users_df = pd.read_csv('data_csv/user.csv')
# user_ids = users_df['id'].tolist()
#
# # Các giá trị gender và tag có thể
# genders = ['male', 'female']
# tags = [1, 3, 4]
#
# # Danh sách job từ category_to_jobs (ví dụ)
# jobs = [
#     'English Instructor', 'Frontend Developer', 'AI Researcher', 'Life Coach',
#     'Graphic Designer', 'Office Administrator', 'Backend Developer',
#     'Data Analyst', 'Business Analyst', 'UX Designer', 'Digital Marketer',
#     'Financial Consultant', 'Fitness Trainer', 'Music Producer',
#     'Photographer', 'Career Advisor'
# ]
#
# # Tạo dữ liệu giả
# overview_data = []
# for i, user_id in enumerate(user_ids, start=6):
#     row = {
#         'id': i,
#         'gender': random.choice(genders),
#         'job': random.choice(jobs),
#         'daily_hours': round(random.uniform(4, 30), 1),
#         'tag': random.choice(tags),
#         'user_id': user_id
#     }
#     overview_data.append(row)
#
# # Chuyển thành DataFrame
# overview_df = pd.DataFrame(overview_data)
#
# # Xuất ra file CSV (nếu muốn)
# overview_df.to_csv('data_csv/overview.csv', index=False)
#
# # In ra một phần dữ liệu mẫu
# print(overview_df.head())

import pandas as pd
import numpy as np


def precision_at_k(recommended, relevant, k=10):
    recommended_at_k = recommended[:k]
    if not relevant:
        return 0.0
    return len(set(recommended_at_k) & set(relevant)) / k


def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return 0.0
    recommended_at_k = recommended[:k]
    return len(set(recommended_at_k) & set(relevant)) / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):
    recommended_at_k = recommended[:k]
    dcg = 0.0
    for i, rec in enumerate(recommended_at_k):
        if rec in relevant:
            dcg += 1 / np.log2(i + 2)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(recommendations_df, ground_truth_df, k=10):
    """
    recommendations_df: DataFrame với các cột ['user_id', 'course_id', 'xgb_score']
    ground_truth_df: DataFrame với các cột ['user_id', 'course_id'], chứa dữ liệu đã tương tác/mua thật
    """
    precisions, recalls, ndcgs = [], [], []

    # Chuyển ground truth thành dictionary: user_id -> list of relevant course_ids
    relevant_dict = ground_truth_df.groupby('user_id')['course_id'].apply(set).to_dict()

    for user_id, group in recommendations_df.groupby('user_id'):
        recommended = group.sort_values('xgb_score', ascending=False)['course_id'].tolist()
        relevant = relevant_dict.get(user_id, set())

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))
        ndcgs.append(ndcg_at_k(recommended, relevant, k))

    return {
        'Precision@{}'.format(k): np.mean(precisions),
        'Recall@{}'.format(k): np.mean(recalls),
        'NDCG@{}'.format(k): np.mean(ndcgs)
    }

