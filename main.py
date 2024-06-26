from flask import Flask, request, render_template
from app_processes import DocumentProcessor, ChainProcessor


# the running codes
data_dir = './data/'
doc_obj = DocumentProcessor(data_dir)
text = doc_obj.readFilez()
chain_obj = ChainProcessor(text)
db_and_chain = chain_obj.CProcessing()

messages = []
app = Flask('__name__', template_folder = './templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        q = request.form['query']
        messages.append({'sender': 'user', 'message': q})
        print(f"q is {q}")

        r = chain_obj.generate_response(q, db_and_chain)
        
        messages.append({'sender': 'bot', 'message': r})

        return render_template('ui_.html',msg = messages)

    return render_template('ui_.html')


if __name__ == '__main__':
  app.run(host='0.0.0.0',port=8080)
