import json

def get_project_config(cfg_file = 'project.json'):
    "This function get general project config values"
    with open("cfg/" + cfg_file, encoding='utf-8') as json_file:
        pcfg = json.load(json_file)
    return pcfg