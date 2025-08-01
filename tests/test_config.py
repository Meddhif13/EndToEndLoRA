import yaml


def test_config_loads():
    with open('config.yaml', 'r') as f:
        data = yaml.safe_load(f)
    assert 'model' in data
