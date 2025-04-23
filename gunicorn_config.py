workers = 1  # Start with 1 worker for stability
worker_class = 'sync'  # Use sync workers instead of gevent
timeout = 120
bind = '0.0.0.0:5000'
keepalive = 5