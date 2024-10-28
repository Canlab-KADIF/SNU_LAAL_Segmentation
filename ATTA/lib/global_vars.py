# global_vars.py
global_mean = None
global_var = None

global_running_mean = None
global_running_var = None

def set_global_mean_var(mean, var):
    global global_mean, global_var
    global_mean = mean
    global_var = var

def get_global_mean_var():
    global global_mean, global_var
    return global_mean, global_var

def set_global_running_mean_var(mean, var):
    global global_running_mean, global_running_var
    global_running_mean = mean
    global_running_var = var

def get_global_running_mean_var():
    global global_running_mean, global_running_var
    return global_running_mean, global_running_var