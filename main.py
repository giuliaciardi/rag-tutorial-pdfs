import subprocess

def run_script(script_name):
    try:
        subprocess.run(['python', script_name], check=True)
        print(f"{script_name} ran successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running {script_name}: {e}")

# List of scripts to run in order
scripts_to_run = [
    'populate_database.py',
    'get_embedding_function.py',
    'main.py',
    'query_data.py',
    'test_rag.py'
]

for script in scripts_to_run:
    run_script(script)
