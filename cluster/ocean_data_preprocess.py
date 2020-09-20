import pandas as pd


def json_to_csv():
    json_path = '../datasets/ocean_news/ocean_news.json'
    csv_path = '../datasets/ocean_news/ocean_news.csv'
    json_data = pd.read_json(json_path, encoding='utf-8')
    json_data.to_csv(csv_path, index=False)


if __name__ == '__main__':
    json_to_csv()