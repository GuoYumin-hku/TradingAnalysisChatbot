# save the upper text analysis into a jsonl file
# keep the structure of dictionary
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d')
        elif isinstance(obj, pd.Period):
            return str(obj)
        return json.JSONEncoder.default(self, obj)

# data loading and preprocessing
data_og = pd.read_csv('../data/merged_data.csv', on_bad_lines='skip', parse_dates=['Order Date', 'Ship Date'])
data_og['Combined_Category'] = data_og['Category'].astype(str) + '-' + data_og['Sub-Category'].astype(str)

def convert_timestamps_to_str(obj):
    if isinstance(obj, dict):
        return {k: convert_timestamps_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_timestamps_to_str(item) for item in obj]
    elif isinstance(obj, pd.Timestamp):
        return obj.strftime('%Y-%m-%d')
    elif isinstance(obj, pd.Period):
        return str(obj)
    return obj

# filter the data
class Analysis:
    def __init__(self, data):
        # Create a deep copy of the data to avoid SettingWithCopyWarning
        self.data = data.copy()
        # Convert date columns to datetime
        self.data['Order Date'] = pd.to_datetime(self.data['Order Date'], errors='coerce')
        self.data['Ship Date'] = pd.to_datetime(self.data['Ship Date'], errors='coerce')
        # Ensure the datetime conversion worked by dropping any rows where the conversion failed
        self.data = self.data.dropna(subset=['Order Date', 'Ship Date'])

    def analyze_segment(self):
        segment_distribution = self.data['Segment'].value_counts().to_dict()
        segment_distribution = {k: (v, round(v / len(self.data), 2)) for k, v in segment_distribution.items()}
        segment_quantity = self.data.groupby('Segment')['Quantity'].mean().round(2).to_dict()
        segment_unit_price = (self.data.groupby('Segment')['Sales'].sum() / self.data.groupby('Segment')['Quantity'].sum()).round(2).to_dict()
        segment_profit = (self.data.groupby('Segment')['Profit'].sum() / self.data.groupby('Segment')['Quantity'].sum()).round(2).to_dict()
        segment_discount = self.data.groupby('Segment')['Discount'].mean().round(2).to_dict()
        return {
            'segment_distribution': segment_distribution,
            'segment_quantity': segment_quantity,
            'segment_unit_price': segment_unit_price,
            'segment_profit': segment_profit,
            'segment_discount': segment_discount
        }
    
    def analyze_region(self):
        region_distribution = self.data['Region'].value_counts().to_dict()
        region_distribution = {k: (v, round(v / len(self.data), 2)) for k, v in region_distribution.items()}
        region_quantity = self.data.groupby('Region')['Quantity'].mean().round(2).to_dict()
        region_sales = self.data.groupby('Region')['Sales'].mean().round(2).to_dict()
        region_unit_price = (self.data.groupby('Region')['Sales'].sum() / self.data.groupby('Region')['Quantity'].sum()).round(2).to_dict()
        region_discount = self.data.groupby('Region')['Discount'].mean().round(2).to_dict()
        region_profit = (self.data.groupby('Region')['Profit'].sum() / self.data.groupby('Region')['Quantity'].sum()).round(2).to_dict()
        region_profit_margin = (self.data.groupby('Region')['Profit'].sum() / self.data.groupby('Region')['Sales'].sum()).round(2).to_dict()
        return {
            'region_distribution': region_distribution,
            'region_quantity': region_quantity,
            'region_sales': region_sales,
            'region_unit_price': region_unit_price,
            'region_discount': region_discount,
            'region_profit': region_profit,
            'region_profit_margin': region_profit_margin
        }
    
    def analyze_market(self):
        market_distribution = self.data['Market'].value_counts().to_dict()
        market_distribution = {k: (v, round(v / len(self.data), 2)) for k, v in market_distribution.items()}
        market_quantity = self.data.groupby('Market')['Quantity'].mean().round(2).to_dict()
        market_sales = self.data.groupby('Market')['Sales'].mean().round(2).to_dict()
        market_unit_price = (self.data.groupby('Market')['Sales'].sum() / self.data.groupby('Market')['Quantity'].sum()).round(2).to_dict()
        market_discount = self.data.groupby('Market')['Discount'].mean().round(2).to_dict()
        market_profit = (self.data.groupby('Market')['Profit'].sum() / self.data.groupby('Market')['Quantity'].sum()).round(2).to_dict()
        market_profit_margin = (self.data.groupby('Market')['Profit'].sum() / self.data.groupby('Market')['Sales'].sum()).round(2).to_dict()
        return {
            'market_distribution': market_distribution,
            'market_quantity': market_quantity,
            'market_sales': market_sales,
            'market_unit_price': market_unit_price,
            'market_discount': market_discount,
            'market_profit': market_profit,
            'market_profit_margin': market_profit_margin
        }
    
    def analyze_country(self):
        country_distribution = self.data['Country'].value_counts().to_dict()
        country_distribution = {k: (v, round(v / len(self.data), 2)) for k, v in country_distribution.items()}
        country_quantity = self.data.groupby('Country')['Quantity'].mean().round(2).to_dict()
        country_sales = self.data.groupby('Country')['Sales'].mean().round(2).to_dict()
        country_unit_price = (self.data.groupby('Country')['Sales'].sum() / self.data.groupby('Country')['Quantity'].sum()).round(2).to_dict()
        country_profit = (self.data.groupby('Country')['Profit'].sum() / self.data.groupby('Country')['Quantity'].sum()).round(2).to_dict()
        country_profit_margin = (self.data.groupby('Country')['Profit'].sum() / self.data.groupby('Country')['Sales'].sum()).round(2).to_dict()
        return {
            'country_distribution': country_distribution,
            'country_quantity': country_quantity,
            'country_sales': country_sales,
            'country_unit_price': country_unit_price,
            'country_profit': country_profit,
            'country_profit_margin': country_profit_margin
        }

    def analyze_state_city(self):
        state_distribution = self.data['State'].value_counts().to_dict()
        state_distribution = {k: (v, round(v / len(self.data), 2)) for k, v in state_distribution.items()}
        top_10_states = dict(list(state_distribution.items())[:10])
        top_10_states_values = [v[0] for v in top_10_states.values()]
        
        city_distribution = self.data['City'].value_counts().to_dict()
        city_distribution = {k: (v, round(v / len(self.data), 2)) for k, v in city_distribution.items()}
        top_10_cities = dict(list(city_distribution.items())[:10])
        top_10_cities_values = [v[0] for v in top_10_cities.values()]
        
        return {
            'top_10_states': top_10_states,
            'top_10_states_values': top_10_states_values,
            'top_10_cities': top_10_cities,
            'top_10_cities_values': top_10_cities_values
        }

    def analyze_product_performance(self):
        product_performance = self.data.groupby('Product Name').agg(
            total_sales=('Sales', 'sum'),
            total_quantity=('Quantity', 'sum'),
            average_price=('Sales', 'mean'),
            total_profit=('Profit', 'sum')
        ).reset_index()

        product_performance['profit_margin'] = product_performance['total_profit']/product_performance['total_sales']
        product_performance = product_performance.sort_values(by='total_sales', ascending=False).head(20)
        return product_performance.to_dict(orient='records')
    
    def analyze_most_ordered_products_by_date(self):
        # Extract year and month from Order Date
        self.data['Order Date'] = pd.to_datetime(self.data['Order Date'])
        self.data['Year_Month'] = self.data['Order Date'].dt.strftime('%Y-%m')
        
        # Group by Year-Month and Product Name
        most_ordered = self.data.groupby(['Year_Month', 'Product Name']).agg(
            total_sales=('Sales', 'sum'),
            total_quantity=('Quantity', 'sum')
        ).reset_index()
        
        # Get top 5 products by total sales
        top_products = most_ordered.groupby('Product Name')['total_sales'].sum().nlargest(5).index
        most_ordered = most_ordered[most_ordered['Product Name'].isin(top_products)]
        
        return most_ordered.to_dict(orient='records')

    def analyze_most_frequent_order_month(self):
        self.data['Order Date'] = pd.to_datetime(self.data['Order Date'], errors='coerce')
        self.data['Order Month'] = self.data['Order Date'].dt.strftime('%Y-%m')

        monthly_orders = self.data['Order Month'].value_counts().reset_index()
        monthly_orders.columns = ['Order Month', 'Order Count']  
        monthly_orders = monthly_orders.sort_values(by='Order Count', ascending=False).head(20)
        
        return monthly_orders.to_dict(orient='records')

    def analyze_time_series(self, date_column):
        self.data.loc[:, date_column] = pd.to_datetime(self.data[date_column])
        
        # Group by Year-Month
        monthly_data = self.data.groupby(self.data[date_column].dt.strftime('%Y-%m')).agg({
            'Sales': 'sum',
            'Quantity': 'sum'
        }).reset_index()
        
        # Rename the date column back
        monthly_data = monthly_data.rename(columns={'index': date_column})
        
        # Calculate 3-month moving average
        monthly_data['Sales_MA'] = monthly_data['Sales'].rolling(window=3).mean()
        
        return monthly_data.to_dict(orient='records')
    
    def analyze_customer_segmentation(self):
        customer_segmentation = self.data.groupby('Customer Name').agg(
            total_spent=('Sales', 'sum'),
            purchase_frequency=('Order ID', 'nunique'),
            average_order_value=('Sales', 'mean')
        ).reset_index()
        # Sort by total spent and get top 10 instead of 20 for better visibility
        customer_segmentation = customer_segmentation.sort_values(by='total_spent', ascending=False).head(10)
        return customer_segmentation.to_dict(orient='records')

    def analyze_discounts(self):
        discount_analysis = self.data.groupby('Discount').agg(
            total_sales=('Sales', 'sum'),
            total_quantity=('Quantity', 'sum'),
            average_profit=('Profit', 'mean')
        ).round(4).reset_index().sort_values('Discount')
        return discount_analysis.to_dict(orient='records')

    def analyze_geographic_distribution(self):
        geographic_distribution = self.data.groupby(['State', 'City']).agg(
            total_sales=('Sales', 'sum'),
            total_quantity=('Quantity', 'sum')
        ).reset_index()
        geographic_distribution['Location'] = geographic_distribution['State'] + ', ' + geographic_distribution['City']
        geographic_distribution = geographic_distribution.sort_values(by='total_sales', ascending=False).head(15)
        return geographic_distribution.to_dict(orient='records')
    
    def analyze_temporal(self):
        # Extract year and month after ensuring dates are datetime
        self.data['Year'] = self.data['Order Date'].dt.year
        self.data['Month'] = self.data['Order Date'].dt.month_name()
        # Calculate shipping delay
        self.data['Shipping_Delay'] = (self.data['Ship Date'] - self.data['Order Date']).dt.days
        
        return {
            'yearly_sales': self.data.groupby('Year')['Sales'].sum().to_dict(),
            'monthly_trend': self.data.groupby(['Year', 'Month'])['Sales'].sum().unstack().to_dict(),
            'shipping_delay_stats': {
                'mean': round(self.data['Shipping_Delay'].mean(), 1),
                'median': self.data['Shipping_Delay'].median(),
                'max': self.data['Shipping_Delay'].max()
            }
        }

    def analyze_products(self):
        product_sales = self.data.groupby('Product Name')['Sales'].sum().nlargest(10)
        product_profit = self.data.groupby('Product Name')['Profit'].sum().nlargest(10)
        product_turnover = self.data.groupby('Product Name')['Quantity'].sum().nlargest(10)
        
        loss_products = self.data[self.data['Profit'] < 0].groupby('Product Name')['Profit'].sum().nsmallest(5)
        
        return {
            'top_10_products_by_sales': product_sales.to_dict(),
            'top_10_products_by_profit': product_profit.to_dict(),
            'top_10_products_by_turnover': product_turnover.to_dict(),
            'top_5_loss_products': loss_products.to_dict()
        }

    def analyze_ship_mode(self):
        analysis = self.data.groupby('Ship Mode').agg(
            total_orders=('Order ID', 'nunique'),
            avg_shipping_delay=('Shipping_Delay', 'mean'),
            total_profit=('Profit', 'sum'),
            shipping_cost_ratio=('Shipping Cost', lambda x: (x / self.data['Sales']).mean())
        ).round(2)
        return analysis.to_dict(orient='index')
    
    def analyze_order_priority(self):
        priority_shipmode = (
            self.data.groupby('Order Priority')[['Ship Mode']].apply(lambda x: x['Ship Mode'].value_counts(normalize=True).to_dict())
        )
        
        profit_margin = (
            self.data.groupby('Order Priority')[['Profit', 'Sales']].apply(lambda x: (x['Profit'] / x['Sales']).mean()).round(2).to_dict()
        )
        
        result = {}
        for priority, shipmode_data in priority_shipmode.items():
            result[priority] = {
                'avg_profit_margin': profit_margin.get(priority, 0),
                'ship_mode_distribution': shipmode_data
            }
        return result
    
    def analyze_rfm(self):
        rfm = self.data.groupby('Customer ID').agg(
            recency=('Order Date', lambda x: (self.data['Order Date'].max() - x.max()).days),
            frequency=('Order ID', 'nunique'),
            monetary=('Sales', 'sum')
        )

        recency_median = rfm['recency'].median()
        rfm['R_Class'] = np.where(rfm['recency'] < recency_median, 'High', 'Low')
        frequency_median = rfm['frequency'].median()
        rfm['F_Class'] = np.where(rfm['frequency'] > frequency_median, 'High', 'Low')
        monetary_median = rfm['monetary'].median()
        rfm['M_Class'] = np.where(rfm['monetary'] > monetary_median, 'High', 'Low')

        rfm['rfm_code'] = (
            rfm['R_Class'].str[0] + rfm['F_Class'].str[0] + rfm['M_Class'].str[0]
        )
        
        group_labels = {
            'HHH': 'VIP',
            'HHL': 'Loyal Customers',
            'HLH': 'Big Spenders',
            'HLL': 'Promising Customers',
            'LHH': 'At-Risk Customers',
            'LHL': 'Slipping Away',
            'LLH': 'Hibernating Customers',
            'LLL': 'Lost Customers'
        }
        rfm['rfm_label'] = rfm['rfm_code'].map(group_labels).fillna('Undefined Group')
        
        return {
            'median_values': {
                'recency': recency_median,
                'frequency': frequency_median,
                'monetary': monetary_median
            },
            'customer_distribution': rfm['rfm_label'].value_counts().to_dict()
        }

    def analyze_churn(self):
        # Calculate RFM metrics
        latest_date = self.data['Order Date'].max()
        
        rfm_data = self.data.groupby('Customer ID').agg({
            'Order Date': lambda x: (latest_date - x.max()).days,  # Recency
            'Order ID': 'count',  # Frequency
            'Sales': 'sum',  # Monetary
            'Discount': 'mean',
            'Quantity': 'mean',
            'Shipping Cost': 'mean',
            'Profit': 'mean'
        }).rename(columns={
            'Order Date': 'Recency',
            'Order ID': 'Frequency',
            'Sales': 'Monetary',
            'Discount': 'Avg_Discount',
            'Quantity': 'Avg_Quantity',
            'Shipping Cost': 'Avg_Shipping_Cost',
            'Profit': 'Avg_Profit'
        })
        
        # Define churn (customers who haven't purchased in 365 days)
        rfm_data['Churn_Status'] = rfm_data['Recency'] > 365
        
        # Calculate churn metrics
        churn_analysis = {
            'total_customers': len(rfm_data),
            'churned_customers': rfm_data['Churn_Status'].sum(),
            'churn_rate': rfm_data['Churn_Status'].mean(),
            'avg_recency': rfm_data['Recency'].mean(),
            'avg_frequency': rfm_data['Frequency'].mean(),
            'avg_monetary': rfm_data['Monetary'].mean(),
            'churned_customer_characteristics': {
                'avg_recency': rfm_data[rfm_data['Churn_Status']]['Recency'].mean(),
                'avg_frequency': rfm_data[rfm_data['Churn_Status']]['Frequency'].mean(),
                'avg_monetary': rfm_data[rfm_data['Churn_Status']]['Monetary'].mean(),
                'avg_discount': rfm_data[rfm_data['Churn_Status']]['Avg_Discount'].mean(),
                'avg_quantity': rfm_data[rfm_data['Churn_Status']]['Avg_Quantity'].mean(),
                'avg_shipping_cost': rfm_data[rfm_data['Churn_Status']]['Avg_Shipping_Cost'].mean(),
                'avg_profit': rfm_data[rfm_data['Churn_Status']]['Avg_Profit'].mean()
            },
            'active_customer_characteristics': {
                'avg_recency': rfm_data[~rfm_data['Churn_Status']]['Recency'].mean(),
                'avg_frequency': rfm_data[~rfm_data['Churn_Status']]['Frequency'].mean(),
                'avg_monetary': rfm_data[~rfm_data['Churn_Status']]['Monetary'].mean(),
                'avg_discount': rfm_data[~rfm_data['Churn_Status']]['Avg_Discount'].mean(),
                'avg_quantity': rfm_data[~rfm_data['Churn_Status']]['Avg_Quantity'].mean(),
                'avg_shipping_cost': rfm_data[~rfm_data['Churn_Status']]['Avg_Shipping_Cost'].mean(),
                'avg_profit': rfm_data[~rfm_data['Churn_Status']]['Avg_Profit'].mean()
            }
        }
        
        return churn_analysis

    def analyze_all(self):
        result = {
            **self.analyze_segment(),
            **self.analyze_region(),
            **self.analyze_market(),
            **self.analyze_country(),
            **self.analyze_state_city(),
            **self.analyze_temporal(),
            **self.analyze_products(),
            'product_performance': self.analyze_product_performance(),  
            'most_ordered_products_by_date': self.analyze_most_ordered_products_by_date(),
            'most_frequent_order_month': self.analyze_most_frequent_order_month(),
            'time_series_analysis': self.analyze_time_series('Order Date'), 
            'customer_segmentation': self.analyze_customer_segmentation(),  
            'discount_analysis': self.analyze_discounts(),  
            'geographic_distribution': self.analyze_geographic_distribution(), 
            'ship_mode_analysis': self.analyze_ship_mode(),
            'order_priority_analysis': self.analyze_order_priority(),
            'rfm_analysis': self.analyze_rfm(),
            'churn_analysis': self.analyze_churn()
        }
        # Convert any remaining timestamps to strings before returning
        return convert_timestamps_to_str(result)

class Visualization:
    def __init__(self, data, analysis_result, category):
        self.data = data.copy()
        self.analysis_result = analysis_result
        self.category = category
        # Convert date columns and calculate shipping delay
        self.data['Order Date'] = pd.to_datetime(self.data['Order Date'], errors='coerce')
        self.data['Ship Date'] = pd.to_datetime(self.data['Ship Date'], errors='coerce')
        # Ensure the datetime conversion worked by dropping any rows where the conversion failed
        self.data = self.data.dropna(subset=['Order Date', 'Ship Date'])
        self.data['Shipping_Delay'] = (self.data['Ship Date'] - self.data['Order Date']).dt.days

    def plot_top_10_states(self):
        top_10_states = self.analysis_result['top_10_states']
        top_10_states_values = self.analysis_result['top_10_states_values']
        
        plt.figure(figsize=(15, 8))
        plt.bar(top_10_states.keys(), top_10_states_values)
        plt.xlabel('State', fontsize=10)
        plt.ylabel('Number of Purchase', fontsize=10)
        plt.title(f'Top 10 States - {self.category}', fontsize=12)
        for i, v in enumerate(top_10_states_values):
            plt.text(i, v, str(v), ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/top_10_states_{self.category}.png')
        plt.close()

    def plot_top_10_cities(self):
        top_10_cities = self.analysis_result['top_10_cities']
        top_10_cities_values = self.analysis_result['top_10_cities_values']
        
        plt.figure(figsize=(15, 8))
        plt.bar(top_10_cities.keys(), top_10_cities_values)
        plt.xlabel('City', fontsize=10)
        plt.ylabel('Number of Purchase', fontsize=10)
        plt.title(f'Top 10 Cities - {self.category}', fontsize=12)
        for i, v in enumerate(top_10_cities_values):
            plt.text(i, v, str(v), ha='center', va='bottom', fontsize=8)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(fontsize=8)
        plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/top_10_cities_{self.category}.png')
        plt.close()
        
    def plot_product_performance(self):
        product_performance = self.analysis_result['product_performance']
        products = [p['Product Name'] for p in product_performance]
        total_sales = [p['total_sales'] for p in product_performance]
        
        plt.figure(figsize=(15, 8))
        plt.bar(range(len(products)), total_sales)
        plt.xlabel('Product Name')
        plt.ylabel('Total Sales')
        plt.title('Product Performance')
        plt.xticks(range(len(products)), products, rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig('./data_analysis/analysis_result_{}/product_performance.png'.format(self.category))
        plt.close()
    
    def plot_customer_segmentation(self):
        customer_data = self.analysis_result['customer_segmentation']
        customers = [c['Customer Name'] for c in customer_data]
        total_spent = [c['total_spent'] for c in customer_data]
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(customers)), total_spent, width=0.7)
        plt.xlabel('Customer Name')
        plt.ylabel('Total Spent ($)')
        plt.title('Top 10 Customers by Total Spent')
        
        # Rotate labels and adjust their position
        plt.xticks(range(len(customers)), customers, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/customer_segmentation.png')
        plt.close()

    def plot_discount_analysis(self):
        discount_data = self.analysis_result['discount_analysis']
        discounts = [d['Discount'] for d in discount_data]
        total_sales = [d['total_sales'] for d in discount_data]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(range(len(discounts)), total_sales, width=0.5)  # Reduced width for more separation
        
        # Format x-axis labels as percentages
        plt.xticks(range(len(discounts)), [f'{d:.0%}' for d in discounts], rotation=45)
        
        # Add grid for better readability
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.xlabel('Discount Rate')
        plt.ylabel('Total Sales ($)')
        plt.title('Sales by Discount Rate')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/discount_analysis.png')
        plt.close()

    def plot_geographic_distribution(self):
        geo_data = self.analysis_result['geographic_distribution']
        locations = [g['Location'] for g in geo_data]
        total_sales = [g['total_sales'] for g in geo_data]
        
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(locations)), total_sales, width=0.7)
        
        plt.xlabel('Location (State, City)')
        plt.ylabel('Total Sales ($)')
        plt.title('Geographic Distribution of Sales')
        plt.xticks(range(len(locations)), locations, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:,.0f}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/geographic_distribution.png')
        plt.close()

    def plot_most_ordered_products_by_date(self):
        most_ordered = self.analysis_result['most_ordered_products_by_date']
        months = [entry['Year_Month'] for entry in most_ordered]
        products = [entry['Product Name'] for entry in most_ordered]
        total_sales = [entry['total_sales'] for entry in most_ordered]

        plt.figure(figsize=(15, 8))
        
        # Create line plot for each unique product
        for product in set(products):
            product_sales = [total_sales[i] for i in range(len(products)) if products[i] == product]
            product_months = [months[i] for i in range(len(products)) if products[i] == product]
            plt.plot(product_months, product_sales, marker='o', linewidth=2, markersize=6, label=product)

        plt.xlabel('Month')
        plt.ylabel('Total Sales ($)')
        plt.title('Top 5 Products Sales Trend')
        
        # Show every nth label to prevent overcrowding
        n = max(len(set(months)) // 12, 1)  # Show about 12 labels
        plt.xticks(range(0, len(set(months)), n), 
                  [months[i] for i in range(0, len(set(months)), n)], 
                  rotation=45, ha='right')
        
        # Move legend outside of plot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Products')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/most_ordered_products_by_date.png',
                   bbox_inches='tight', dpi=300)
        plt.close()

    def plot_time_series(self):
        time_series_data = self.analysis_result['time_series_analysis']
        dates = [pd.to_datetime(ts['Order Date']).strftime('%Y-%m') for ts in time_series_data]
        total_sales = [ts['Sales'] for ts in time_series_data]
        sales_ma = [ts['Sales_MA'] for ts in time_series_data]
        
        plt.figure(figsize=(15, 8))
        
        # Plot actual sales
        plt.plot(dates, total_sales, marker='o', markersize=4, alpha=0.6, label='Monthly Sales')
        
        # Plot moving average
        plt.plot(dates, sales_ma, linewidth=2, color='red', label='3-Month Moving Average')
        
        plt.xlabel('Month')
        plt.ylabel('Total Sales ($)')
        plt.title('Sales Trend Over Time')
        
        # Show every nth label to prevent overcrowding
        n = max(len(dates) // 12, 1)  # Show about 12 labels
        plt.xticks(range(0, len(dates), n), [dates[i] for i in range(0, len(dates), n)],
                  rotation=45, ha='right')
        
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/time_series_sales.png')
        plt.close()

    def plot_temporal_analysis(self):
        plt.figure(figsize=(12, 6))
        yearly = self.analysis_result['yearly_sales']
        plt.plot(list(yearly.keys()), list(yearly.values()), marker='o')
        plt.title(f'Yearly Sales Trend - {self.category}')
        plt.xlabel('Year')
        plt.ylabel('Total Sales')
        plt.grid(True)
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/yearly_trend_{self.category}.png')
        plt.close()

        monthly = pd.DataFrame(self.analysis_result['monthly_trend'])
        plt.figure(figsize=(14, 8))
        sns.heatmap(monthly, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title(f'Monthly Sales Heatmap - {self.category}')
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/monthly_heatmap_{self.category}.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.hist(self.data['Shipping_Delay'].dropna(), bins=20, edgecolor='black')
        plt.title(f'Shipping Delay Distribution - {self.category}')
        plt.xlabel('Days')
        plt.ylabel('Frequency')
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/shipping_delay_{self.category}.png')
        plt.close()

    def plot_product_analysis(self):
        self._plot_product_ranking('top_10_products_by_sales', 'Sales', 'Total Sales')
        self._plot_product_ranking('top_10_products_by_profit', 'Profit', 'Total Profit')
        self._plot_product_ranking('top_10_products_by_turnover', 'Turnover', 'Total Quantity Sold')
        
        self._plot_loss_products()

    def _plot_product_ranking(self, result_key, metric, ylabel):
        data = self.analysis_result[result_key]
        if not data:
            return
    
        products = list(data.keys())
        values = list(data.values())
        
        cumulative = np.cumsum(values) / sum(values) * 100
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        ax1.bar(products, values, alpha=0.5)
        ax1.set_xlabel('Product Name', fontsize=12)
        ax1.set_ylabel(ylabel, fontsize=12)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        
        ax2 = ax1.twinx()
        ax2.plot(products, cumulative, color='r', marker='o', linewidth=2)
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax2.grid(False)
        
        for i, (v, c) in enumerate(zip(values, cumulative)):
            ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
            ax2.text(i, c, f'{c:.1f}%', ha='center', va='bottom', color='red', fontsize=8)
        
        plt.title(f'Pareto Analysis - {metric} - {self.category}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/pareto_{metric.lower()}_{self.category}.png')
        plt.close()

    def _plot_loss_products(self):
        data = self.analysis_result['top_5_loss_products']
        if not data:
            return

        plt.figure(figsize=(12, 6))
        plt.bar(data.keys(), data.values(), color='red')
        plt.title(f'Top 5 Loss-Making Products - {self.category}')
        plt.xlabel('Product Name')
        plt.ylabel('Total Loss')
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/top_loss_products_{self.category}.png')
        plt.close()
        
    def plot_ship_mode_analysis(self):
        data = self.analysis_result['ship_mode_analysis']
        
        save_dir = f'./data_analysis/analysis_result_{self.category}/'
        os.makedirs(save_dir, exist_ok=True)
        
        pie_colors = sns.color_palette("pastel", len(data))
        
        plt.figure(figsize=(10, 8))
        costs = [v['shipping_cost_ratio'] for v in data.values()]
        plt.pie(
            costs,
            labels=data.keys(),
            autopct='%1.1f%%',
            colors=pie_colors,
            startangle=90,
            wedgeprops={'edgecolor': 'white', 'linewidth': 1}
        )
        plt.title('Shipping Cost Ratio Distribution', fontsize=14, pad=20)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}shipping_cost_ratio.png', dpi=300)
        plt.close()
        
    def plot_order_priority_analysis(self):
        data = self.analysis_result['order_priority_analysis']
        
        plt.figure(figsize=(12, 6))
        priorities = list(data.keys())
        margins = [v['avg_profit_margin'] for v in data.values()]
        plt.bar(priorities, margins, color='teal')
        plt.title('Average Profit Margin by Order Priority')
        plt.ylabel('Profit Margin Ratio')
        plt.xticks(rotation=45)
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/order_priority_margin.png')
        plt.close()
        
        all_ship_modes = set()
        for priority_data in data.values():
            all_ship_modes.update(priority_data['ship_mode_distribution'].keys())
        all_ship_modes = sorted(all_ship_modes)
        
        priorities = list(data.keys())
        shipmode_percentages = {mode: [] for mode in all_ship_modes}
        
        for priority in priorities:
            dist = data[priority]['ship_mode_distribution']
            for mode in all_ship_modes:
                shipmode_percentages[mode].append(dist.get(mode, 0))
        
        plt.figure(figsize=(12, 6))
        bottom = np.zeros(len(priorities))
        
        for mode in all_ship_modes:
            percentages = shipmode_percentages[mode]
            plt.bar(
                priorities, 
                percentages, 
                label=mode, 
                bottom=bottom,
                edgecolor='white'
            )
            bottom += np.array(percentages)
        
        plt.title('Transport Mode Distribution by Order Priority')
        plt.xlabel('Order Priority')
        plt.ylabel('Percentage')
        plt.legend(title='Ship Mode', bbox_to_anchor=(1.05, 1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/priority_shipmode_distribution.png')
        plt.close()
        
    def plot_rfm_analysis(self):
        rfm = self.data.groupby('Customer Name').agg(
            recency=('Order Date', lambda x: (self.data['Order Date'].max() - x.max()).days),
            frequency=('Order ID', 'nunique'),
            monetary=('Sales', 'sum')
        )
        
        plt.figure(figsize=(14, 8))
        ax = plt.subplot(projection='3d')
        ax.scatter(rfm['recency'], rfm['frequency'], rfm['monetary'], 
                   c=rfm['monetary'], cmap='viridis', s=50)
        ax.set_xlabel('Recency (Days)')
        ax.set_ylabel('Frequency')
        ax.set_zlabel('Monetary ($)')
        plt.title('3D RFM Analysis')
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/rfm_3d.png')
        plt.close()
        
        segments = self.analysis_result['rfm_analysis']['customer_distribution']
        plt.figure(figsize=(10, 6))
        plt.pie(segments.values(), labels=segments.keys(), autopct='%1.1f%%')
        plt.title('Customer Activity Segmentation')
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/rfm_segments.png')
        plt.close()

    def plot_churn_analysis(self):
        churn_data = self.analysis_result['churn_analysis']
        
        # Plot 1: Churn Rate
        plt.figure(figsize=(10, 6))
        plt.pie([1 - churn_data['churn_rate'], churn_data['churn_rate']], 
                labels=['Active', 'Churned'],
                autopct='%1.1f%%',
                colors=['lightgreen', 'lightcoral'])
        plt.title('Customer Churn Rate')
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/churn_rate.png')
        plt.close()
        
        # Plot 2: Comparison of Characteristics
        metrics = ['avg_recency', 'avg_frequency', 'avg_monetary', 'avg_discount', 
                  'avg_quantity', 'avg_shipping_cost', 'avg_profit']
        churned_values = [churn_data['churned_customer_characteristics'][m] for m in metrics]
        active_values = [churn_data['active_customer_characteristics'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        plt.figure(figsize=(15, 8))
        plt.bar(x - width/2, active_values, width, label='Active Customers', color='lightgreen')
        plt.bar(x + width/2, churned_values, width, label='Churned Customers', color='lightcoral')
        
        plt.xlabel('Metrics')
        plt.ylabel('Values')
        plt.title('Comparison of Active vs Churned Customer Characteristics')
        plt.xticks(x, [m.replace('avg_', '').replace('_', ' ').title() for m in metrics], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/churn_characteristics.png')
        plt.close()

    def plot_most_frequent_order_month(self):
        monthly_orders = self.analysis_result['most_frequent_order_month']
        order_months = [entry['Order Month'] for entry in monthly_orders]
        order_counts = [entry['Order Count'] for entry in monthly_orders]

        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(order_months)), order_counts, width=0.7)
        
        plt.xlabel('Order Month')
        plt.ylabel('Number of Orders')
        plt.title('Order Frequency by Month')
        
        # Rotate labels and adjust their position
        plt.xticks(range(len(order_months)), order_months, rotation=45, ha='right')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:,.0f}',
                    ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'./data_analysis/analysis_result_{self.category}/most_frequent_order_month.png')
        plt.close()

for y1 in set(data_og['Combined_Category']):
    print(f'Processing category: {y1}')
    dir_path = f'./data_analysis/analysis_result_{y1}'
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    # Create a deep copy of the filtered data
    data = data_og[data_og['Combined_Category'] == y1].copy()
    
    analysis = Analysis(data)
    analysis_result = analysis.analyze_all()
    
    with open(f'{dir_path}/analysis_result_{y1}.json', 'w') as f:
        json.dump(analysis_result, f, indent=4, cls=NumpyEncoder)
    
    visualization = Visualization(data, analysis_result, y1)
    visualization.plot_top_10_states()
    visualization.plot_top_10_cities()
    visualization.plot_temporal_analysis()
    visualization.plot_product_analysis()
    visualization.plot_product_performance()
    visualization.plot_most_ordered_products_by_date()
    visualization.plot_most_frequent_order_month()
    visualization.plot_time_series()
    visualization.plot_customer_segmentation()
    visualization.plot_discount_analysis()
    visualization.plot_geographic_distribution()
    visualization.plot_ship_mode_analysis()
    visualization.plot_order_priority_analysis()
    visualization.plot_rfm_analysis()
    visualization.plot_churn_analysis()
