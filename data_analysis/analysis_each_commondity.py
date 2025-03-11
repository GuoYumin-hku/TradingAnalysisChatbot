# save the upper text analysis into a jsonl file
# keep the structure of dictionary
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# data loading and preprocessing
data_og = pd.read_csv('../data/merged_data.csv', on_bad_lines='skip')
data_og['Combined_Category'] = data_og['Category'].astype(str) + '-' + data_og['Sub-Category'].astype(str)

# filter the data
class Analysis:
    def __init__(self, data):
        self.data = data
        self.data.loc[:, 'Order_Date'] = pd.to_datetime(self.data['Order Date'])
        self.data.loc[:, 'Ship_Date'] = pd.to_datetime(self.data['Ship Date'])
        

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
        product_performance = self.data.groupby('Product ID').agg(
            total_sales=('Sales', 'sum'),
            total_quantity=('Quantity', 'sum'),
            average_price=('Sales', 'mean'),
            total_profit=('Profit', 'sum')
        ).reset_index()

        product_performance['profit_margin'] = product_performance['total_profit']/product_performance['total_sales']
        product_performance = product_performance.sort_values(by='total_sales', ascending=False).head(20)
        return product_performance.to_dict(orient='records')
    
    def analyze_most_ordered_products_by_date(self):
        most_ordered = self.data.groupby(['Order Date', 'Product Name']).agg(
            total_sales=('Sales', 'sum'),
            total_quantity=('Quantity', 'sum')
        ).reset_index()
        
        most_ordered = most_ordered.sort_values(by=['Order Date', 'total_sales'], ascending=[True, False]).head(20)
        
        return most_ordered.to_dict(orient='records')  

    def analyze_most_frequent_order_month(self):
        self.data['Order Date'] = pd.to_datetime(self.data['Order Date'], errors='coerce')
        self.data['Order Month'] = self.data['Order Date'].dt.to_period('M').astype(str)

        monthly_orders = self.data['Order Month'].value_counts().reset_index()
        monthly_orders.columns = ['Order Month', 'Order Count']  
        monthly_orders = monthly_orders.sort_values(by='Order Count', ascending=False).head(20)
        
        return monthly_orders.to_dict(orient='records') 

    def analyze_time_series(self, date_column):
        self.data.loc[:, date_column] = pd.to_datetime(self.data[date_column], errors='coerce')
        #if self.data[date_column].isnull().all():
         #   raise ValueError(f"All values in '{date_column}' could not be converted to datetime.")
        
        #if not pd.api.types.is_datetime64_any_dtype(self.data[date_column]):
         #   raise TypeError(f"The column '{date_column}' is not in datetime format.")
    
        time_series_analysis = self.data.groupby(self.data[date_column].dt.to_period('M')).agg(
           total_sales=('Sales', 'sum'),
          total_quantity=('Quantity', 'sum')
        ).reset_index()
        
        time_series_analysis[date_column] = time_series_analysis[date_column].astype(str)
        return time_series_analysis.to_dict(orient='records')
    
    def analyze_customer_segmentation(self):
        customer_segmentation = self.data.groupby('Customer ID').agg(
            total_spent=('Sales', 'sum'),
            purchase_frequency=('Order ID', 'nunique'),
            average_order_value=('Sales', 'mean')
        ).reset_index()
        customer_segmentation = customer_segmentation.sort_values(by='total_spent', ascending=False).head(20)
        return customer_segmentation.to_dict(orient='records')

    def analyze_discounts(self):
        discount_analysis = self.data.groupby('Discount').agg(
            total_sales=('Sales', 'sum'),
            total_quantity=('Quantity', 'sum'),
            average_profit=('Profit', 'mean')
        ).reset_index()
        return discount_analysis.to_dict(orient='records')

    def analyze_geographic_distribution(self):
        geographic_distribution = self.data.groupby(['State', 'City']).agg(
            total_sales=('Sales', 'sum'),
            total_quantity=('Quantity', 'sum')
        ).reset_index()
        geographic_distribution = geographic_distribution.sort_values(by='total_sales', ascending=False).head(20)
        return geographic_distribution.to_dict(orient='records')
    
    def analyze_temporal(self):
        self.data.loc[:, 'Year'] = self.data['Order_Date'].dt.year
        self.data['Month'] = self.data['Order_Date'].dt.month_name()
        self.data.loc[:, 'Shipping_Delay'] = (self.data['Ship_Date'] - self.data['Order_Date']).dt.days
        
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

    def analyze_all(self):
        return {
            **self.analyze_segment(),
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
        }


for y1 in list(set(data_og['Combined_Category'])):
    print(y1)
    if not os.path.exists('./analysis_result_{}'.format(y1)):
        os.makedirs('./analysis_result_{}'.format(y1))
    data = data_og[data_og['Combined_Category'] == y1]
    analysis = Analysis(data)
    analysis_result = analysis.analyze_all()
    with open('./analysis_result_{}/analysis_result_{}.json'.format(y1, y1), 'w') as f:
        json.dump(analysis_result, f, indent=4)

class Visualization:
    def __init__(self, data, analysis_result, category):
        self.data = data
        self.analysis_result = analysis_result
        self.category = category

    def plot_top_10_states(self):
        top_10_states = self.analysis_result['top_10_states']
        top_10_states_values = self.analysis_result['top_10_states_values']
        
        plt.figure(figsize=(12, 6))
        plt.bar(top_10_states.keys(), top_10_states_values)
        plt.xlabel('State')
        plt.ylabel('Number of Purchase')
        plt.title(f'Top 10 States - {self.category}')
        for i, v in enumerate(top_10_states_values):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.xticks(rotation=45)
        plt.savefig(f'./analysis_result_{self.category}/top_10_states_{self.category}.png')
        plt.close()

    def plot_top_10_cities(self):
        top_10_cities = self.analysis_result['top_10_cities']
        top_10_cities_values = self.analysis_result['top_10_cities_values']
        
        plt.figure(figsize=(12, 6))
        plt.bar(top_10_cities.keys(), top_10_cities_values)
        plt.xlabel('City')
        plt.ylabel('Number of Purchase')
        plt.title(f'Top 10 Cities - {self.category}')
        for i, v in enumerate(top_10_cities_values):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.xticks(rotation=45)
        plt.savefig(f'./analysis_result_{self.category}/top_10_cities_{self.category}.png')
        plt.close()
        
    def plot_product_performance(self):
            product_performance = self.analysis_result['product_performance']
            products = [p['Product ID'] for p in product_performance]
            total_sales = [p['total_sales'] for p in product_performance]
            plt.figure(figsize=(12, 6))
            plt.bar(products, total_sales)
            plt.xlabel('Product ID')
            plt.ylabel('Total Sales')
            plt.title('Product Performance')
            plt.xticks(rotation=45)
            plt.savefig('./analysis_result_{}/product_performance.png'.format(self.category))
            plt.close()
    
    def plot_most_ordered_products_by_date(self):
        most_ordered = self.analysis_result['most_ordered_products_by_date']
        dates = [entry['Order Date'] for entry in most_ordered]
        products = [entry['Product Name'] for entry in most_ordered]
        total_sales = [entry['total_sales'] for entry in most_ordered]

        plt.figure(figsize=(12, 6))
        for product in set(products):
            product_sales = [total_sales[i] for i in range(len(products)) if products[i] == product]
            product_dates = [dates[i] for i in range(len(products)) if products[i] == product]
            plt.plot(product_dates, product_sales, marker='o', label=product)

        plt.xlabel('Order Date')
        plt.ylabel('Total Sales')
        plt.title('Most Ordered Products by Date')
        plt.xticks(rotation=45)
        plt.legend(title='Product Name')
        plt.tight_layout()
        plt.savefig('./analysis_result_{}/most_ordered_products_by_date.png'.format(self.category))
        plt.close()

    def plot_most_frequent_order_month(self):
        monthly_orders = self.analysis_result['most_frequent_order_month']
        order_months = [entry['Order Month'] for entry in monthly_orders]
        order_counts = [entry['Order Count'] for entry in monthly_orders]

        plt.figure(figsize=(12, 6))
        plt.bar(order_months, order_counts, color='skyblue')
        plt.xlabel('Order Month')
        plt.ylabel('Order Count')
        plt.title('Most Frequent Order Month')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('./analysis_result_{}/most_frequent_order_month.png'.format(self.category))
        plt.close()
    
    def plot_time_series(self):
        time_series_data = self.analysis_result['time_series_analysis']
        months = [str(ts['Order Date']) for ts in time_series_data]
        total_sales = [ts['total_sales'] for ts in time_series_data]
        plt.figure(figsize=(12, 6))
        plt.plot(months, total_sales, marker='o')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.title('Sales Over Time')
        plt.xticks(rotation=45)
        plt.savefig('./analysis_result_{}/time_series_sales.png'.format(self.category))
        plt.close()
    
    def plot_customer_segmentation(self):
        customer_data = self.analysis_result['customer_segmentation']
        customers = [c['Customer ID'] for c in customer_data]
        total_spent = [c['total_spent'] for c in customer_data]
        plt.figure(figsize=(12, 6))
        plt.bar(customers, total_spent)
        plt.xlabel('Customer ID')
        plt.ylabel('Total Spent')
        plt.title('Customer Segmentation')
        plt.xticks(rotation=45)
        plt.savefig('./analysis_result_{}/customer_segmentation.png'.format(self.category))
        plt.close()

    def plot_discount_analysis(self):
        discount_data = self.analysis_result['discount_analysis']
        discounts = [d['Discount'] for d in discount_data]
        total_sales = [d['total_sales'] for d in discount_data]
        plt.figure(figsize=(12, 6))
        plt.bar(discounts, total_sales)
        plt.xlabel('Discount')
        plt.ylabel('Total Sales')
        plt.title('Sales by Discount')
        plt.xticks(rotation=45)
        plt.savefig('./analysis_result_{}/discount_analysis.png'.format(self.category))
        plt.close()

    def plot_geographic_distribution(self):
        geo_data = self.analysis_result['geographic_distribution']
        states = [f"{g['State']}, {g['City']}" for g in geo_data]
        total_sales = [g['total_sales'] for g in geo_data]
        plt.figure(figsize=(12, 6))
        plt.bar(states, total_sales)
        plt.xlabel('Location (State, City)')
        plt.ylabel('Total Sales')
        plt.title('Geographic Distribution of Sales')
        plt.xticks(rotation=45)
        plt.savefig('./analysis_result_{}/geographic_distribution.png'.format(self.category))
        plt.close()   

    def plot_temporal_analysis(self):
        plt.figure(figsize=(12, 6))
        yearly = self.analysis_result['yearly_sales']
        plt.plot(list(yearly.keys()), list(yearly.values()), marker='o')
        plt.title(f'Yearly Sales Trend - {self.category}')
        plt.xlabel('Year')
        plt.ylabel('Total Sales')
        plt.grid(True)
        plt.savefig(f'./analysis_result_{self.category}/yearly_trend_{self.category}.png')
        plt.close()

        monthly = pd.DataFrame(self.analysis_result['monthly_trend'])
        plt.figure(figsize=(14, 8))
        sns.heatmap(monthly, annot=True, fmt=".0f", cmap="YlGnBu")
        plt.title(f'Monthly Sales Heatmap - {self.category}')
        plt.savefig(f'./analysis_result_{self.category}/monthly_heatmap_{self.category}.png')
        plt.close()

        plt.figure(figsize=(10, 6))
        self.data['Shipping_Delay'].hist(bins=20)
        plt.title(f'Shipping Delay Distribution - {self.category}')
        plt.xlabel('Days')
        plt.ylabel('Frequency')
        plt.savefig(f'./analysis_result_{self.category}/shipping_delay_{self.category}.png')
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
        
        plt.xticks(rotation=45, ha='right')
        
        ax2 = ax1.twinx()
        ax2.plot(products, cumulative, color='r', marker='o', linewidth=2)
        ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
        ax2.grid(False)
        
        for i, (v, c) in enumerate(zip(values, cumulative)):
            ax1.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontsize=9)
            ax2.text(i, c, f'{c:.1f}%', ha='center', va='bottom', color='red')
        
        plt.title(f'Pareto Analysis - {metric} - {self.category}', fontsize=14)
        plt.tight_layout()
        plt.savefig(f'./analysis_result_{self.category}/pareto_{metric.lower()}_{self.category}.png')
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
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'./analysis_result_{self.category}/top_loss_products_{self.category}.png')
        plt.close()

for y1 in set(data_og['Combined_Category']):
    print(f'Processing category: {y1}')
    dir_path = f'./analysis_result_{y1}'
    
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    data = data_og.loc[data_og['Combined_Category'] == y1].copy()
    
    analysis = Analysis(data)
    analysis_result = analysis.analyze_all()
    
    with open(f'{dir_path}/analysis_result_{y1}.json', 'w') as f:
        json.dump(analysis_result, f, indent=4)
    
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
