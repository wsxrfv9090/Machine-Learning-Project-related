import os

def choose_work_directory(project_name):
    current_dir = os.getcwd()
    if os.path.basename(current_dir) == "Machine-Learning-Project-related":
        project_dir_ABS = os.path.join(current_dir, project_name)
        os.makedirs(project_dir_ABS, exist_ok = True)
        data_dir = os.path.join(project_dir_ABS, 'Data')
        os.makedirs(data_dir, exist_ok = True)
        output_dir = os.path.join(project_dir_ABS, 'Output')
        os.makedirs(output_dir, exist_ok = True)
    else:
        print(f'Current Working Directory: ' + current_dir)
    
choose_work_directory('test project 1')