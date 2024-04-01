import importlib.util
import os

class ModelManager():
    def __init__(self, models_path = "./models"):
        if os.path.isabs(models_path):
            self.models_path = models_path
        else:
            dir_name = os.path.dirname(__file__)
            self.models_path = os.path.join(dir_name, models_path)
    
        self.models = {}
        model_dirs = os.listdir(self.models_path)
        
        for model_dir in model_dirs:
            model_path = self.models_path + "/" + model_dir + "/model.py"
            
            spec = importlib.util.spec_from_file_location("models." + model_dir, model_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            model = module.Model
            self.models[model.get_name()] = model
            
            
    def get_model_names(self):
        return list(self.models.keys())
        
        
    def get_model_info(self, model_name):
        return self.models[model_name].get_description()

        
    def create_model(self, name):            
        return self.models[name]()


model_manager = ModelManager()
