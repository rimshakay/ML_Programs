import os
import sqlite3
from flask import Flask, render_template, request, url_for, flash, redirect
from sqlalchemy import false
from werkzeug.exceptions import abort
import DecisionTreeRegressor.simpleimputer as dtree
import RandomForestRegressor.simpleimputer as rforest
import pandas as pd

def get_db_connection():
    conn=sqlite3.connect('database.db')
    conn.row_factory=sqlite3.Row
    return conn

def get_model(model_id):
    conn = get_db_connection()
    model = conn.execute('SELECT * FROM model WHERE id = ?',
                        (model_id,)).fetchone()
    conn.close()
    if model is None:
        abort(404)
    return model

app = Flask(__name__)
app.config['SECRET_KEY']='myregressions'

@app.route('/')
def index():
    conn=get_db_connection()
    models=conn.execute('SELECT * FROM model').fetchall()
    conn.close()
    return render_template('index.html',models=models)

@app.route('/<int:model_id>', methods=('GET', 'POST'))
def model(model_id):
    model = get_model(model_id)
    print("entered")

    result="result"
    if request.method == 'GET':
        df=pd.DataFrame()
        if model_id==2:
            print(model_id)
            rforest.checkSales()
            df=pd.read_csv('data/store-sales-time-series-forecasting/random_forest_predictions.csv')
            result=df.to_string(index=False)
    conn = get_db_connection()
    conn.execute('UPDATE model SET predictions = ?'
                    ' WHERE id = ?',
                    (result, model_id,))
    conn.commit()
    conn.close()
    model = get_model(model_id)
    
    #     #     return redirect(url_for('index'))
    return render_template('model.html', model=model)

# @app.route('/create', methods=('GET', 'POST'))
# def create():
#     if request.method == 'POST':
#         title = request.form['title']
#         content = request.form['content']

#         if not title:
#             flash('Title is required!')
#         else:
#             conn = get_db_connection()
#             conn.execute('INSERT INTO history (model,params, predictions) VALUES (?, ?, ?)',
#                          (title, content))
#             conn.commit()
#             conn.close()
#             return redirect(url_for('index'))

#     return render_template('create.html')

# @app.route('/<int:id>/edit', methods=('GET', 'POST'))
# def edit(id):
#     model = get_model(id)

#     if request.method == 'POST':
#         title = request.form['title']
#         content = request.form['content']

#         if not title:
#             flash('Title is required!')
#         else:
#             conn = get_db_connection()
#             conn.execute('UPDATE history SET title = ?, content = ?'
#                          ' WHERE id = ?',
#                          (title, content, id))
#             conn.commit()
#             conn.close()
#             return redirect(url_for('index'))

#     return render_template('edit.html', model=model)

# @app.route('/<int:id>/delete', methods=('POST',))
# def delete(id):
#     model = get_model(id)
#     conn = get_db_connection()
#     conn.execute('DELETE FROM history WHERE id = ?', (id,))
#     conn.commit()
#     conn.close()
#     flash('"{}" was successfully deleted!'.format(model['title']))
#     return redirect(url_for('index'))

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)
