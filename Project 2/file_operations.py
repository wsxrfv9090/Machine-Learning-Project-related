import pandas as pd
import os
repo_name = "Machine-Learning-Project-related"

def find_repo_base(current_dir, repo_name):
    parts = current_dir.split(os.sep)
    if repo_name in parts:
        index = parts.index(repo_name)
        repo_base = os.sep.join(parts[:index + 1])
        if os.name == 'nt' and ':' in parts[0]:
            repo_base = parts[0] + os.sep + os.sep.join(parts[1:index + 1])
        return repo_base
    else:
        return None

def create_project_path(project_name):
    current_dir = os.getcwd()
    repo_base = find_repo_base(current_dir, repo_name)
    
    if repo_base:
        project_dir = os.path.join(repo_base, project_name)
        os.makedirs(project_dir, exist_ok = True)
        print(f"Project directory created under '{project_dir}'")
        
        data_dir = os.path.join(project_dir, 'Data')
        os.makedirs(data_dir, exist_ok = True)
        print(f"Project data directory created under '{data_dir}'")
        
        output_dir = os.path.join(project_dir, 'Output')
        os.makedirs(output_dir, exist_ok = True)
        print(f"Project output directory created under '{output_dir}'")
        return {
            'project_path': project_dir,
            'data_dir': data_dir,
            'output_dir': output_dir,
        }
    else:
        print(f"WARNING: The repository '{repo_name}' was not found in the current directory path ({current_dir}).")
        return None



def read_and_return_pd_df(file_path):
    print("The input file path is: ")
    print(file_path)
    print("Reading the input file...")

    # Check if the file is csv or excel
    if file_path.endswith('.csv'):
        input_file_pd = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        input_file_pd = pd.read_excel(file_path)
    else:
        print("The file format is not supported. Please provide a csv or excel file.")
        return False
    return input_file_pd
    
def change_head_to_ENG(pd_df):
    # Change the column names to English
    # 证券代码  证券名称    交易时间	开盘价	最高价	最低价	收盘价	涨跌	涨跌幅	成交量	成交额
    pd_df.columns = ['SECU_CODE', 'SECU_NAME', 'DATE', 'OPENING', 'HIGHEST', 'LOWEST', 'CLOSING', 'CHANGE', 'PCT_CHANGE', 'VOLUME', 'AMOUNT']

def change_secu_code_to_str(pd_df):
    # Change the data type of the security code to string
    pd_df['SECU_CODE'] = pd_df['SECU_CODE'].astype('U6')
    
def change_date_to_datetime(pd_df):
    # Change the data type of the date to datetime
    pd_df['DATE'] = pd.to_datetime(pd_df['DATE'])
    
def change_numerical_data_to_float64(pd_df):
    # Change the data type of the numerical data to float64
    cols = ['OPENING', 'HIGHEST', 'LOWEST', 'CLOSING', 'CHANGE', 'PCT_CHANGE', 'VOLUME', 'AMOUNT']
    pd_df[cols] = pd_df[cols].replace({',': '', '--': 'NaN'}, regex = True).astype('float64')