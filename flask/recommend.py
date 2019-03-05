
from flask import Flask
from flask import Flask, render_template, flash, request, redirect, url_for
from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms import RadioField
import pickle
import pandas as pd
from flask_table import Table, Col
import numpy as np
import math

from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy

from surprise import Dataset
from surprise import accuracy
from surprise.model_selection import KFold
from surprise import NormalPredictor

from surprise import Dataset
from surprise import Reader
from surprise.model_selection import cross_validate
from surprise.model_selection import GridSearchCV

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import ColumnDataSource, NumeralTickFormatter
from bokeh.palettes import Spectral6, Pastel1
from bokeh.models import ColumnDataSource
from bokeh.plotting import show, output_notebook, figure as bf
from bokeh.plotting import figure
from bokeh.models.glyphs import  Text
from bokeh.layouts import row, widgetbox
from bokeh.models import CustomJS, Slider
from bokeh.io import curdoc
from bokeh.layouts import row, column
from bokeh.models.widgets import Slider, TextInput
from bokeh.models import HoverTool
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource, ranges, LabelSet

from sklearn.metrics.pairwise import cosine_similarity
 
# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'
# Declare your table
class ItemTable(Table):
    rating = Col('My Rating')
    title = Col('Title')
    average_rating = Col('Average Goodreads Rating')
class Item(object):
    def __init__(self, rating, title):
        self.rating = rating
        self.title = title
        self.average_rating = average_rating

class ReusableForm(Form):
    name = TextField('Name:', validators=[validators.required()])
class NameForm(Form):
    title = StringField('Title of book', validators=[validators.required()])
    submit = SubmitField('Submit')
class SimpleForm(Form):
    radio = RadioField('Theme', choices=['young adult','epic fantasy/fairytale','supernatural','sci-fi','urban fantasy'])
    submit1 = SubmitField('Refresh')

#with open("user_product_matrix.pkl", "rb") as f:
#    mat = pickle.load(f)
#with open("user_product_long.pkl", "rb") as f2:
#    df = pickle.load(f2)
df = pd.read_pickle('user_product_long_v2.pkl')
df = df[df['rating'] != 0]
with open("recommender_predictions.pkl", "rb") as f3:
    recs = pickle.load(f3)
with open("book_titles_map.pkl", 'rb') as f4:
    fantasy_books = pickle.load(f4)
with open("recommender_algorithm.pkl", "rb") as f:
    algo = pickle.load(f)
book_rating_aggs = pd.read_pickle('fantasy_books_info_dataframe.pkl')
book_topics = pd.read_pickle('final_book_topics.pkl')
topic_dict = {'topic_1':'young adult',
              'topic_2':'epic fantasy/fairytale',
              'topic_3':'supernatural',
              'topic_4':'sci-fi',
              'topic_5':'urban fantasy'
             }
df = pd.merge(df, book_topics[['bookid','topic_1','topic_2','topic_3','topic_4','topic_5','topic','topic_name']], how='left', on='bookid')

@app.route("/", methods=['GET', 'POST'])
def hello():
    form = ReusableForm(request.form)
    name = ''
    print (form.errors)
    if request.method == 'POST' and form.validate():
        userid = request.form.get('name')
        return redirect(url_for('get_recs', userid=userid))
    else:
        return render_template('main_page.html', form=form, name=name)

@app.route('/results', methods=["POST", "GET"])
def get_content(bookid):
    title=bookid
    #title=request.args.get('title')
    #title = df['title'][df['bookid'] == title]

    cols = ['topic_1','topic_2','topic_3','topic_4','topic_5']
    matrix = book_topics[cols].values
    def get_books(title):
        i = book_topics[book_topics['title']==title].index[0] # i is index corresponding to that title      
        cosines = cosine_similarity(matrix[i:i+1], matrix)
        cosines = cosines.tolist()[0]
        sim_book_indexes = sorted(range(len(cosines)), key=lambda i: cosines[i])[-20:]
            #cosines = sorted(cosines, key=lambda x: x[1], reverse=True)
        titles = []
        for idx in sim_book_indexes:
            if book_topics['title'].iloc[idx] == title:
                pass
            else:
                titles.append(book_topics['title'].iloc[idx])
        for t in titles:
            for n in range(2,20):
                if '#'+str(n) in t:
                    try:
                        titles.remove(t)
                    except:
                        pass
            #return (" \n".join(map(str,titles)))
        return(set(titles))

    #results = get_books(title)
    print (title)
    return render_template('content_results.html', title=title)

@app.route('/<string:userid>', methods=["POST", "GET"])
def get_recs(userid):

    cols = ['topic_1','topic_2','topic_3','topic_4','topic_5']
    matrix = book_topics[cols].values
    def get_books(title):
        i = book_topics[book_topics['title']==title].index[0] # i is index corresponding to that title      
        cosines = cosine_similarity(matrix[i:i+1], matrix)
        cosines = cosines.tolist()[0]
        sim_book_indexes = sorted(range(len(cosines)), key=lambda i: cosines[i])[-20:]
            #cosines = sorted(cosines, key=lambda x: x[1], reverse=True)
        titles = []
        for idx in sim_book_indexes:
            if book_topics['title'].iloc[idx] == title:
                pass
            else:
                titles.append(book_topics['title'].iloc[idx])
        for t in titles:
            for n in range(2,20):
                if '#'+str(n) in t:
                    titles.remove(t)
            #return (" \n".join(map(str,titles)))
        return(set(titles))

    title = ''
    bookid=''
    results = ''
    content_form = NameForm(request.form)
    #if request.form.get('title') == 'submit1':
    if request.method == 'POST' and content_form.validate:
        #if request.form.get('title') == 'content_form':
            #book_title = request.form['title']
        if content_form.submit:
            title = request.form.get('title')
            bookid = df['bookid'][df['title'] == title]  
            results = get_books(title) 
        elif topic_form.submit1:
            pass
        #flash('Searched for: %s' % request.form.get('title') %bookid)
        #return redirect(url_for('get_content', title=request.form.get('title')))


    #10015197

    ### User=specific content ###
    books_read = df[['rating','title','bookid','topic_1','topic_2','topic_3','topic_4','topic_5']][df['userid']==int(userid)].drop_duplicates()
    
    books_read = pd.merge(books_read, book_rating_aggs[['bookid','average_rating']], on = 'bookid',how='left')
    books_read = books_read.sort_values(by='rating', ascending = False)
    books_not_read = list(set(df['bookid'].unique()) - set(list(books_read)))
    ### User topics ###
    data = pd.DataFrame(books_read[['topic_1','topic_2','topic_3','topic_4','topic_5']].mean()).reset_index()
    favorite_topic = data.max()['index']
    data.columns=['topic','values']
    data['topic'] = data['topic'].map(topic_dict)
    data['values'] = [X*100 for X in data['values']]
    data = data.sort_values(by='values', ascending = False)
    colors = ['orangered','tomato','coral','darkorange','orange']
    source = ColumnDataSource(data=dict(x=data['values'], y=data['topic'], colors=colors))
    plot = figure(y_range = (0,data['values'].max()+5), x_range = source.data['y'], plot_height=200,plot_width=500)
    plot.vbar(x = 'y', bottom=0, top='x', width=0.6, color='colors', source=source)
    plot.add_tools(HoverTool(tooltips=[("topic:", "@y"), ("Percent:", "@x{int}%")]))
    #plot.xaxis.axis_label = 'Sub-genre'
    plot.xaxis.major_label_orientation = math.pi/4
    plot.xaxis.major_label_text_font_size = "8pt"
    plot.xaxis.axis_label_text_font_size = "8pt"
    plot.xaxis.axis_label_text_font_style = "italic"
    plot.yaxis.axis_label = 'Percent read'
    plot.yaxis.minor_tick_line_color = None
    plot.yaxis.major_label_text_font_size = "8pt"
    plot.yaxis.axis_label_text_font_size = "8pt"
    plot.yaxis.axis_label_text_font_style = "italic"
    plot.xaxis.major_label_text_font_style = "bold"
    plot.xgrid.grid_line_color = None
    plot.ygrid.grid_line_color = None
    plot.yaxis.major_tick_line_color = None
    plot.xaxis.major_tick_line_color = None
    plot.background_fill_alpha = 0
    plot.border_fill_alpha = 0
    
    ###
    estimates = []
    for bookid in books_not_read:
        uid = int(userid)
        iid = bookid 
        estimates.append(algo.predict(uid, iid, verbose=False))

    book_recs = pd.DataFrame(estimates).sort_values(by='est', ascending = False)
    book_recs['title'] = book_recs['iid'].map(fantasy_books)
    book_recs['est'] = [round(X, 2) for X in book_recs['est']]

    book_recs = book_recs[(book_recs['est'] == 5.0)]
    for i in range(2,20):
        book_recs = book_recs[book_recs['title'].str.contains('#'+str(i))==False]
    book_recs = pd.merge(book_recs, 
                     book_rating_aggs[['bookid','average_rating','ratings_count']],
                     left_on = 'iid',
                     right_on = 'bookid',
                     how = 'left'
                    )   
    book_recs = pd.merge(book_recs, book_topics[['bookid','topic_1',
                                                'topic_2','topic_3',
                                                'topic_4','topic_5','topic','topic_name']],
                                                on='bookid', how='left' )
    book_recs['log_rating_rank'] = book_recs['average_rating']*np.log(book_recs['ratings_count']*book_recs[favorite_topic])
    book_recs = book_recs.sort_values(by='log_rating_rank', ascending = False)
    

    ###
    items = []
    for index, row in books_read.iterrows():
        row_dict = {}
        row_dict['rating'] = (u'\u2606'*int(row['rating']))
        row_dict['title'] = row['title']
        row_dict['average_rating'] = row['average_rating']
        items.append(row_dict)
    table = ItemTable(items)
    print(table.__html__())
    
    items2=[]
    #topic_name = request.form.get("themes")
    #book_recs = book_recs[book_recs['topic_name'] == topic_name]
    for index, row in book_recs.iterrows():
        row_dict = {}
        row_dict['rating'] = row['est']
        row_dict['title'] = row['title']
        row_dict['average_rating'] = row['average_rating']
        items2.append(row_dict)
    table2 = ItemTable(items2[0:25])
    print(table2.__html__())

    javascript, div = components(plot)


    boxes = list(df['topic_name'][(df['topic_name'].isna()==False) & (df['topic_name']!='None')].unique())
    topic_form = SimpleForm(request.form)
    #if request.method == 'POST' and topic_form.validate:
    #    if request.form.post['themes'] == 'sci-fi':
        
    #    if content_form.submit:
    #        pass
    #    elif topic_form.submit1:
            #if 'topic_form' in request.form:
        #book_title = request.form['title']
    #        topic_name = 'sci-fi'

        #bookid = df['bookid'][df['title'] == title]  
        #results = get_books(title) 
        #flash('Searched for: %s' % request.form.get('title') %bookid)
        #return redirect(url_for('get_content', title=request.form.get('title')))
    #if form1.submit1.data and form1.validate():
    #    topic_name = request.form1.get("themes")
    #    book_recs = book_recs[book_recs['topic_name'] == topic_name]#

    #    items2 = []
    #    for index, row in book_recs.iterrows():
    #        row_dict = {}
    #        row_dict['rating'] = row['est']
    #        row_dict['title'] = row['title']
    #        row_dict['average_rating'] = row['average_rating']
    #        items2.append(row_dict)
        # Populate the table
    ##    table2 = ItemTable(items2[0:25])
     #   print(table2.__html__())
        #return render_template('user_page.html', boxes=boxes, title=title, bookid=bookid, results=results, javascript=javascript, div=div, name=userid, books_read=books_read, table=table, table2=table2)

    #layout = row(age_dist, s)   
    #curdoc().add_root(plot)
    

    return render_template('user_page.html', content_form=content_form, topic_form=topic_form, boxes=boxes, title=title, bookid=bookid, results=results, javascript=javascript, div=div, name=userid, books_read=books_read, table=table, table2=table2)

    def books_read(userid):
        print ("Books Read:")
    
    def books_recs(userid):
        print("Recommendations for "+userid+":")
    #return redirect('/')
 
if __name__ == "__main__":
    app.run(debug=True)