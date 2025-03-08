# save the upper text analysis into a jsonl file
# keep the structure of dictionary
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# data loading and preprocessing
data_og = pd.read_csv('../data/merged_data.csv', on_bad_lines='skip')
data_og['Combined_Category'] = data_og['Category'].astype(str) + '-' + data_og['Sub-Category'].astype(str)

# filter the data
class Analysis:
    def __init__(self, data):
        self.data = data

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
        product_performance = self.data.groupby('Product_ID').agg(
            total_sales=('Sales', 'sum'),
            total_quantity=('Quantity', 'sum'),
            average_price=('Sales', 'mean'),
            total_profit=('Profit', 'sum')
        ).reset_index()

        product_performance['profit_margin'] = product_performance['total_profit']/product_performance['total_sales']
        return product_performance.to_dict(orient='records')
    
    def analyze_time_series(self, date_column):
        self.data[date_column] = pd.to_datetime(self.data[date_column], errors='coerce')
        time_series_analysis = self.data.groupby(self.data[date_column].dt.to_period('M')).agg(
            total_sales=('Sales', 'sum'),
            total_quantity=('Quantity', 'sum')
        ).reset_index()
        return time_series_analysis.to_dict(orient='records')
    
    def analyze_customer_segmentation(self):
        customer_segmentation = self.data.groupby('Customer_ID').agg(
            total_spent=('Sales', 'sum'),
            purchase_frequency=('Order_ID', 'nunique'),
            average_order_value=('Sales', 'mean')
        ).reset_index()
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
        return geographic_distribution.to_dict(orient='records')
    
    def analyze_returns(self):
        # Assuming there is a 'Return' column indicating if an item was returned
        return_analysis = self.data[self.data['Return'] == 1].groupby('Product_ID').agg(
            total_returns=('Return', 'count'),
            total_sales_lost=('Sales', 'sum')
        ).reset_index()
        return return_analysis.to_dict(orient='records')
    
    def analyze_all(self):
        return {
            **self.analyze_segment(),
            **self.analyze_country(),
            **self.analyze_state_city(),
            **self.analyze_product_performance(),
            **self.analyze_time_series('Order_Date'),
            **self.analyze_customer_segmentation(),
            **self.analyze_discounts(),
            **self.analyze_geographic_distribution(),
            **self.analyze_returns()
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
        plt.title('Top 10 state distribution of item {}'.format(self.category))
        for i, v in enumerate(top_10_states_values):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.xticks(rotation=45)
        plt.savefig('./analysis_result_{}/top_10_states_distribution_{}.png'.format(self.category, self.category))

    def plot_top_10_cities(self):
        top_10_cities = self.analysis_result['top_10_cities']
        top_10_cities_values = self.analysis_result['top_10_cities_values']
        plt.figure(figsize=(12, 6))
        plt.bar(top_10_cities.keys(), top_10_cities_values)
        plt.xlabel('City')
        plt.ylabel('Number of Purchase')
        plt.title('Top 10 city distribution of item {}'.format(self.category))
        for i, v in enumerate(top_10_cities_values):
            plt.text(i, v, str(v), ha='center', va='bottom')
        plt.xticks(rotation=45)
        plt.savefig('./analysis_result_{}/top_10_cities_distribution_{}.png'.format(self.category, self.category))

    def plot_product_performance(self):
        product_performance = self.analysis_result['product_performance']
        products = [p['Product_ID'] for p in product_performance]
        total_sales = [p['total_sales'] for p in product_performance]
        plt.figure(figsize=(12, 6))
        plt.bar(products, total_sales)
        plt.xlabel('Product ID')
        plt.ylabel('Total Sales')
        plt.title('Product Performance')
        plt.xticks(rotation=45)
        plt.savefig('./analysis_result_{}/product_performance.png'.format(self.category))
        plt.close()

    def plot_time_series(self):
        time_series_data = self.analysis_result['time_series_analysis']
        months = [str(ts['Order_Date']) for ts in time_series_data]
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
        customers = [c['Customer_ID'] for c in customer_data]
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
    
    def plot_returns_analysis(self):
        returns_data = self.analysis_result['returns_analysis']
        products = [r['Product_ID'] for r in returns_data]
        total_returns = [r['total_returns'] for r in returns_data]
        plt.figure(figsize=(12, 6))
        plt.bar(products, total_returns)
        plt.xlabel('Product ID')
        plt.ylabel('Total Returns')
        plt.title('Returns Analysis')
        plt.xticks(rotation=45)
        plt.savefig('./analysis_result_{}/returns_analysis.png'.format(self.category))
        plt.close()

for y1 in list(set(data_og['Combined_Category'])):
    print(y1)
    if not os.path.exists('./analysis_result_{}'.format(y1)):
        os.makedirs('./analysis_result_{}'.format(y1))
    data = data_og[data_og['Combined_Category'] == y1]
    analysis = Analysis(data)
    analysis_result = analysis.analyze_all()
    with open('./analysis_result_{}/analysis_result_{}.json'.format(y1, y1), 'w') as f:
        json.dump(analysis_result, f, indent=4)
    visualization = Visualization(data, analysis_result, y1)
    visualization.plot_top_10_states()
    visualization.plot_top_10_cities()
    visualization.plot_product_performance()
    visualization.plot_time_series()
    visualization.plot_customer_segmentation()
    visualization.plot_discount_analysis()
    visualization.plot_geographic_distribution()
    visualization.plot_returns_analysis()
