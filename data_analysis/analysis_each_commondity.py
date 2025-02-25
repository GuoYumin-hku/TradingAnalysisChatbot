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


    def analyze_all(self):
        return {
            **self.analyze_segment(),
            **self.analyze_country(),
            **self.analyze_state_city()
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
