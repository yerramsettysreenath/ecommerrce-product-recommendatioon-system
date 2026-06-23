import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
df = pd.read_csv(r'C:\Users\HP\Downloads\ecommerce_dataset_500.csv')
df['combined_features'] = df['product_name'] + " " + df['category'] + " " + df['description']
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['combined_features'])
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster_id'] = kmeans.fit_predict(tfidf_matrix)
def recommend_using_kmeans(target_product, num_recommendations=5):
    all_products = df['product_name'].str.lower().tolist()
    if target_product.lower() in all_products:
        
        product_row = df[df['product_name'].str.lower() == target_product.lower()].iloc[0]
        
        assigned_cluster = product_row['cluster_id']
        
        same_cluster_products = df[(df['cluster_id'] == assigned_cluster) & (df['product_name'].str.lower() != target_product.lower()) ]
        
        recommendations = same_cluster_products[['product_name', 'category', 'price', 'rating']].head(num_recommendations)
        
        print(recommendations.to_string(index=False))
    else:
        print(f"\nSorry, '{target_product}' is not in the dataset.")
userinput=input("enter the product name ")
recommend_using_kmeans(userinput)