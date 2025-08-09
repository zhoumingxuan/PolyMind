import json

class ConfigHelper:
    def __init__(self):
        self.config_data = self._load_config()

    def _load_config(self):
        """加载配置文件并返回数据"""
        try:
            with open("setting.json", 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"配置文件 setting.json 不存在！")
            return {}
        except json.JSONDecodeError:
            print("配置文件格式错误！")
            return {}

    def get(self, key, default=None):
        """获取配置值，如果不存在则返回默认值"""
        return self.config_data.get(key, default)

