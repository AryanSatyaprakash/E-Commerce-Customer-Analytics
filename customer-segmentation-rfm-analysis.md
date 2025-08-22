# Customer Segmentation with RFM Analysis

This notebook demonstrates a comprehensive RFM (Recency, Frequency, Monetary) analysis for customer segmentation using the e-commerce transaction data.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Loading & Preparation](#data-preparation)
3. [RFM Analysis](#rfm-analysis)
   - [Recency Calculation](#recency)
   - [Frequency Calculation](#frequency)
   - [Monetary Value Calculation](#monetary)
4. [Customer Segmentation](#customer-segmentation)
   - [K-Means Clustering](#kmeans)
   - [Segment Interpretation](#interpretation)
5. [Visualization](#visualization)
6. [Business Recommendations](#recommendations)

## 1. Introduction <a name="introduction"></a>

RFM (Recency, Frequency, Monetary) analysis is a customer segmentation technique that uses past purchase behavior to divide customers into groups. By examining:
- **Recency**: How recently a customer made a purchase
- **Frequency**: How often they purchase
- **Monetary Value**: How much they spend

We can create targeted marketing strategies for different customer segments.

## 2. Data Loading & Preparation <a name="data-preparation"></a>

```python
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import warnings
warnings.filterwarnings('ignore')

# Set plot aesthetics
plt.style.use('seaborn-whitegrid')
sns.set(font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)

# Load preprocessed data
df = pd.read_csv('data/processed/cleaned_transactions.csv')

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"Timeframe: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")
print(f"Number of transactions: {df['InvoiceNo'].nunique():,}")
print(f"Number of customers: {df['CustomerID'].nunique():,}")
print(f"Number of products: {df['StockCode'].nunique():,}")

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Filter for positive quantities (exclude returns/cancellations)
df_purchases = df[df['Quantity'] > 0]

# Display sample data
df_purchases.head()
```

## 3. RFM Analysis <a name="rfm-analysis"></a>

### 3.1 Recency Calculation <a name="recency"></a>

```python
# Get the most recent purchase date for analysis
max_date = df_purchases['InvoiceDate'].max()
print(f"Most recent date in the dataset: {max_date}")

# Add a day to make sure we capture the most recent purchases correctly
max_date = max_date + pd.Timedelta(days=1)

# Create RFM table
rfm = df_purchases.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'Revenue': 'sum'  # Monetary
})

# Rename columns
rfm.columns = ['Recency', 'Frequency', 'Monetary']

# Display RFM table
print(f"RFM table shape: {rfm.shape}")
rfm.head()
```

### 3.2 Frequency Calculation <a name="frequency"></a>

```python
# Analyze frequency distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=rfm, x='Frequency', bins=50, kde=True)
plt.title('Distribution of Purchase Frequency')
plt.xlabel('Number of Purchases')
plt.ylabel('Count of Customers')
plt.axvline(rfm['Frequency'].median(), color='r', linestyle='--', 
            label=f'Median: {rfm["Frequency"].median()}')
plt.legend()
plt.show()

# Identify high-frequency customers
high_freq_customers = rfm[rfm['Frequency'] > rfm['Frequency'].quantile(0.9)]
print(f"Number of high-frequency customers: {len(high_freq_customers)}")
print(f"Percentage of total customers: {len(high_freq_customers) / len(rfm) * 100:.2f}%")
```

### 3.3 Monetary Value Calculation <a name="monetary"></a>

```python
# Analyze monetary distribution
plt.figure(figsize=(12, 6))
sns.histplot(data=rfm, x='Monetary', bins=50, kde=True)
plt.title('Distribution of Customer Spending (Monetary)')
plt.xlabel('Total Spending (£)')
plt.ylabel('Count of Customers')
plt.axvline(rfm['Monetary'].median(), color='r', linestyle='--', 
            label=f'Median: £{rfm["Monetary"].median():.2f}')
plt.legend()
plt.show()

# Handle outliers for better visualization
rfm_filtered = rfm[rfm['Monetary'] < rfm['Monetary'].quantile(0.99)]

plt.figure(figsize=(12, 6))
sns.histplot(data=rfm_filtered, x='Monetary', bins=50, kde=True)
plt.title('Distribution of Customer Spending (Without Outliers)')
plt.xlabel('Total Spending (£)')
plt.ylabel('Count of Customers')
plt.axvline(rfm_filtered['Monetary'].median(), color='r', linestyle='--', 
            label=f'Median: £{rfm_filtered["Monetary"].median():.2f}')
plt.legend()
plt.show()
```

## 4. Customer Segmentation <a name="customer-segmentation"></a>

### 4.1 K-Means Clustering <a name="kmeans"></a>

```python
# Create a copy of RFM data for scaling
rfm_for_clustering = rfm.copy()

# Log transform to handle skewed data
rfm_for_clustering['Recency_log'] = np.log1p(rfm_for_clustering['Recency'])
rfm_for_clustering['Frequency_log'] = np.log1p(rfm_for_clustering['Frequency'])
rfm_for_clustering['Monetary_log'] = np.log1p(rfm_for_clustering['Monetary'])

# Scale the data
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_for_clustering[['Recency_log', 'Frequency_log', 'Monetary_log']])

# Find optimal number of clusters using Elbow method
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2, 10))
visualizer.fit(rfm_scaled)
optimal_k = visualizer.elbow_value_
visualizer.show()

print(f"Optimal number of clusters: {optimal_k}")

# Apply K-Means with optimal k
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
rfm_for_clustering['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Add cluster labels back to original RFM data
rfm['Cluster'] = rfm_for_clustering['Cluster']

# Show cluster distribution
cluster_counts = rfm['Cluster'].value_counts().sort_index()
print("\nCluster Distribution:")
for cluster, count in cluster_counts.items():
    print(f"Cluster {cluster}: {count} customers ({count/len(rfm)*100:.2f}%)")

# Analyze cluster characteristics
cluster_analysis = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'median', 'count']
}).round(2)

print("\nCluster Characteristics:")
print(cluster_analysis)
```

### 4.2 Segment Interpretation <a name="interpretation"></a>

```python
# Define a function to assign segment labels based on cluster characteristics
def assign_segment_labels(cluster_data):
    segments = {}
    
    # Get average values for each cluster
    for cluster in cluster_data['Cluster'].unique():
        cluster_stats = cluster_data[cluster_data['Cluster'] == cluster].mean()
        
        # Assign labels based on RFM values
        if cluster_stats['Recency'] < 30 and cluster_stats['Frequency'] > 5 and cluster_stats['Monetary'] > 1000:
            segments[cluster] = "Champions"
        elif cluster_stats['Recency'] < 90 and cluster_stats['Frequency'] > 3 and cluster_stats['Monetary'] > 500:
            segments[cluster] = "Loyal Customers"
        elif cluster_stats['Recency'] < 60 and cluster_stats['Frequency'] <= 3:
            segments[cluster] = "Potential Loyalists"
        elif cluster_stats['Recency'] >= 60 and cluster_stats['Recency'] < 180 and cluster_stats['Frequency'] > 2:
            segments[cluster] = "At Risk Customers"
        elif cluster_stats['Recency'] >= 180 and cluster_stats['Frequency'] > 2:
            segments[cluster] = "Hibernating"
        elif cluster_stats['Recency'] < 30 and cluster_stats['Frequency'] == 1:
            segments[cluster] = "New Customers"
        elif cluster_stats['Recency'] >= 180 and cluster_stats['Frequency'] == 1:
            segments[cluster] = "Lost Customers"
        else:
            segments[cluster] = "Others"
    
    return segments

# Assign segment labels
segments = assign_segment_labels(rfm)
segment_mapping = {k: f"Segment {k}: {v}" for k, v in segments.items()}

# Add segment labels to RFM data
rfm['Segment'] = rfm['Cluster'].map(segments)

# Display segment distribution
segment_counts = rfm['Segment'].value_counts()
print("\nCustomer Segment Distribution:")
for segment, count in segment_counts.items():
    print(f"{segment}: {count} customers ({count/len(rfm)*100:.2f}%)")

# Analyze segments in more detail
segment_analysis = rfm.groupby('Segment').agg({
    'Recency': ['mean', 'median'],
    'Frequency': ['mean', 'median', 'max'],
    'Monetary': ['mean', 'median', 'sum']
}).round(2)

print("\nSegment Detailed Analysis:")
print(segment_analysis)
```

## 5. Visualization <a name="visualization"></a>

```python
# Create a 3D scatter plot of RFM values by segment
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Color map for segments
colors = plt.cm.tab10(np.linspace(0, 1, len(rfm['Segment'].unique())))
segment_colors = {segment: colors[i] for i, segment in enumerate(rfm['Segment'].unique())}

# Plot each segment
for segment, color in segment_colors.items():
    segment_data = rfm[rfm['Segment'] == segment]
    ax.scatter(
        segment_data['Recency'],
        segment_data['Frequency'],
        segment_data['Monetary'],
        c=[color],
        s=50,
        alpha=0.6,
        label=segment
    )

ax.set_xlabel('Recency (days)')
ax.set_ylabel('Frequency (purchases)')
ax.set_zlabel('Monetary (£)')
ax.set_title('3D RFM Segmentation')
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

# Create radar chart for segment comparison
from math import pi
import matplotlib.pyplot as plt

# Prepare the data
segment_profiles = rfm.groupby('Segment').mean().reset_index()

# Scale the values between 0 and 1 for radar chart
for col in ['Recency', 'Frequency', 'Monetary']:
    if col == 'Recency':  # Lower is better for recency
        segment_profiles[f'{col}_scaled'] = 1 - (segment_profiles[col] / segment_profiles[col].max())
    else:  # Higher is better for frequency and monetary
        segment_profiles[f'{col}_scaled'] = segment_profiles[col] / segment_profiles[col].max()

# Set data
categories = ['Recency', 'Frequency', 'Monetary']
N = len(categories)

# Create the radar chart
fig = plt.figure(figsize=(12, 8))
for i, segment in enumerate(segment_profiles['Segment']):
    # Convert 0-1 scale to angles
    values = segment_profiles.loc[i, [f'{c}_scaled' for c in categories]].values.tolist()
    values += values[:1]  # Close the loop
    
    # Calculate angle for each category
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Create subplot
    ax = plt.subplot(2, 3, i+1, polar=True)
    
    # Draw segment line
    ax.plot(angles, values, linewidth=2, linestyle='solid', label=segment)
    ax.fill(angles, values, alpha=0.1)
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title(segment, size=14)
    
plt.tight_layout()
plt.show()
```

## 6. Business Recommendations <a name="recommendations"></a>

Based on our RFM analysis and customer segmentation, here are strategic recommendations for each segment:

### Champions
- **Profile**: Recent purchases, frequent buyers, high spending
- **Strategy**: Reward and engage
- **Actions**:
  - Implement loyalty rewards program
  - Offer exclusive early access to new products
  - Consider them for product testing and feedback
  - Maintain regular communication without overwhelming

### Loyal Customers
- **Profile**: Regular purchases with good spending
- **Strategy**: Upsell and retain
- **Actions**:
  - Create personalized product recommendations
  - Offer bundle discounts on favorite product categories
  - Implement "next level" spending incentives
  - Engage with satisfaction surveys

### Potential Loyalists
- **Profile**: Recent customers with moderate spending
- **Strategy**: Build relationship and increase frequency
- **Actions**:
  - Offer targeted discounts on second purchase
  - Create email onboarding series highlighting product range
  - Provide exceptional customer service
  - Send educational content about product usage

### At Risk Customers
- **Profile**: Haven't purchased recently but were active previously
- **Strategy**: Reactivate and prevent churn
- **Actions**:
  - Send "We miss you" campaigns with special offers
  - Ask for feedback on previous purchases
  - Showcase new products related to previous purchases
  - Offer incentives tied to returning within 30 days

### Hibernating
- **Profile**: Long period of inactivity but multiple past purchases
- **Strategy**: Carefully reconnect
- **Actions**:
  - Send "comeback" offers with significant incentives
  - Highlight major product improvements or new lines
  - Consider multi-channel outreach (email, direct mail, SMS)
  - Ask for feedback on why they stopped purchasing

### New Customers
- **Profile**: Single recent purchase
- **Strategy**: Create second purchase opportunity
- **Actions**:
  - Send first-time buyer follow-up emails
  - Offer complementary products to initial purchase
  - Provide helpful content related to purchased product
  - Ask for reviews to increase engagement

### Lost Customers
- **Profile**: Single purchase long ago
- **Strategy**: Learn and potentially reactivate
- **Actions**:
  - Survey to understand why they didn't return
  - Test win-back campaigns with strong offers
  - Update with major brand or product changes
  - Consider removing from regular marketing if no response

## Implementation Planning

1. **Short-term Actions (1-2 months):**
   - Set up automated email campaigns for each segment
   - Create segment-specific promotions for the next campaign
   - Develop a dashboard to track segment performance

2. **Medium-term Actions (3-6 months):**
   - Implement a loyalty program for Champions and Loyal Customers
   - Develop personalized product recommendation system
   - Create segment migration tracking to measure effectiveness

3. **Long-term Strategy (6-12 months):**
   - Build predictive churn models for At Risk identification
   - Implement CLV (Customer Lifetime Value) prediction by segment
   - Develop omnichannel personalization strategy

## Next Steps

- Re-segment customers quarterly to track movement between segments
- A/B test different offers for each segment to optimize response rates
- Integrate website behavior data to enhance segmentation accuracy
- Develop machine learning models to predict segment migration