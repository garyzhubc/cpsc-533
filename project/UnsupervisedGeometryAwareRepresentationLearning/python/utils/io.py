import os, sys, time, shutil, importlib
import importlib.util

def savePythonFile(config_orig_path, directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    config_save_path = '{}/{}'.format(directory_path, os.path.basename(config_orig_path))
<<<<<<< HEAD
    if not os.path.exists(config_save_path):
        shutil.copy(config_orig_path, config_save_path)
        print('copying {} to {}'.format(config_orig_path, config_save_path))
=======
    # if not os.path.exists(config_save_path):
    shutil.copy(config_orig_path, config_save_path)
    print('copying {} to {}'.format(config_orig_path, config_save_path))
>>>>>>> 3da462b9351869b0342b95d99fef37ab3e45a309

def loadModule(module_path_and_name):
    # if contained in module it would be a oneliner: 
    # config_dict_module = importlib.import_module(dict_module_name) 
    module_child_name = module_path_and_name.split('/')[-1].replace('.py','')
    spec = importlib.util.spec_from_file_location(module_child_name, module_path_and_name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
<<<<<<< HEAD
    return module
=======
    return module
>>>>>>> 3da462b9351869b0342b95d99fef37ab3e45a309
