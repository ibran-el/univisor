from flask import Flask, request, render_template, session
from app_processes import DocumentProcessor, ChainProcessor
from dotenv import load_dotenv
import os

# the running codes
data_dir = './data/'
doc_obj = DocumentProcessor(data_dir)
text = doc_obj.readFilez()
chain_obj = ChainProcessor(text)
doc_and_chain = chain_obj.CProcessing()

load_dotenv()

app = Flask('__name__', template_folder = './templates')
secret_key = os.getenv("SECRET_KEY")
app.config['SECRET_KEY'] = secret_key
# messages = []

@app.route('/', methods=['GET', 'POST'])
def index():
    if 'messages' not in session:
        session['messages'] = []  # Initialize session messages list if not present

    if request.method == 'POST':
        q = request.form['query']
        q+="\. Make your response neat and easy to understand"
        session['messages'].append({'sender': 'user', 'message': q})

        r = chain_obj.generate_response(q, doc_and_chain)
        
        session['messages'].append({'sender': 'bot', 'message': r})

        return render_template('ui_.html',msg = session['messages'])

    return render_template('ui_.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
