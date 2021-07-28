# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 10:04:10 2021

@author: Lenovo
"""

from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World"

@app.route("/harry")
def harry():
    return "Hello Harry Bhai"

app.run(debug = True)