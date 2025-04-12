# Configuration file for jupyter-notebook

c = get_config()

# Network settings
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False
c.NotebookApp.allow_origin = '*'

# Security settings - Disable password and token authentication
c.NotebookApp.password = ''
c.NotebookApp.token = ''
c.NotebookApp.allow_password_change = False

# Interface settings
c.NotebookApp.terminado_settings = {'shell_command': ['/bin/bash']}

# Set the notebook directory
c.NotebookApp.notebook_dir = '/workspaces/StayAwake-AI/notebooks'
