import sqlite3

connection = sqlite3.connect('database.db')


with open('schema.sql') as f:
    connection.executescript(f.read())

cur = connection.cursor()

cur.execute("INSERT INTO model (model,params, predictions) VALUES (?,?,?)",
            ('Decision Tree Regressor','','pred')
            )

cur.execute("INSERT INTO model (model,params, predictions) VALUES (?,?,?)",
            ('Random Forest Regressor','','pred')
            )

connection.commit()
connection.close()
