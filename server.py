import cherrypy
from main import app  # Assuming your Flask app is defined in app.py

if __name__ == '__main__':
    cherrypy.tree.graft(app, "/")  # Mount the Flask app at the root
    cherrypy.config.update({
        'server.socket_host': '0.0.0.0',  # Listen on all available network interfaces
        'server.socket_port': 8080,  # Port to listen on
    })
    cherrypy.engine.start()
    cherrypy.engine.block()
