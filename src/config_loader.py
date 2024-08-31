import yaml

def load_config(filepath='config.yaml'):
    try:
        with open(filepath, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return None