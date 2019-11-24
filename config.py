import yaml
from box import Box


def config():
    with open('./yml/config.yml', 'r') as f:                                           
        data = yaml.load(stream=f, Loader=yaml.SafeLoader)

    data = Box(data)
    return data