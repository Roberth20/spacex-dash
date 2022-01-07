# Import required libraries
import pandas as pd
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output
import plotly.express as px


# Read the airline data into pandas dataframe
spacex_df = pd.read_csv("spacex_launch_dash.csv")
max_payload = spacex_df['Payload Mass (kg)'].max()
min_payload = spacex_df['Payload Mass (kg)'].min()

# Create a dash application
app = dash.Dash(__name__)

# Create an app layout
app.layout = html.Div(children=[html.H1('SpaceX Launch Records Dashboard',
                                        style={'textAlign': 'center', 'color': '#503D36',
                                               'font-size': 40}),
                                # TASK 1: Add a dropdown list to enable Launch Site selection
                                # The default select value is for ALL sites
                                dcc.Dropdown(id='site-dropdown', 
                                             options=[{"label":"All Sites", "value":"ALL"},
                                                      {"label":"CCAFS LC-40", "value":"CCAFS LC-40"},
                                                      {"label":"VAFB SLC-4E", "value":"VAFB SLC-4E"},
                                                      {"label":"KSC LC-39A", "value":"KSC LC-39A"},
                                                      {"label":"CCAFS SLC-40", "value":"CCAFS SLC-40"}],
                                             value="ALL",
                                             placeholder="Select a Launch Site",
                                             searchable=True),
                                html.Br(),

                                # TASK 2: Add a pie chart to show the total successful launches count for all sites
                                # If a specific launch site was selected, show the Success vs. Failed counts for the site
                                html.Div(dcc.Graph(id='success-pie-chart')),
                                html.Br(),

                                html.P("Payload range (Kg):"),
                                # TASK 3: Add a slider to select payload range
                                dcc.RangeSlider(id='payload-slider',
                                                min=0,
                                                max=10000,
                                                step=1000,
                                                marks={0:{"label":"0"},
                                                      2500:{"label":"2500"},
                                                      5000:{"label":"5000"},
                                                      10000:{"label":"10000"}},
                                                value=[min_payload, max_payload]),

                                # TASK 4: Add a scatter chart to show the correlation between payload and launch success
                                html.Div(dcc.Graph(id='success-payload-scatter-chart')),
                               # Seccion de Machine Learning y modelos
                                html.H2("Best performer model"),
                                # Seleccion de modelos
                                dcc.Dropdown(id="model-option",
                                            options=[{"label":"Logistic Regression", 
                                                      "value":"logreg"},
                                                    {"label":"Support Vector Model",
                                                    "value":"svm"},
                                                    {"label":"Decision Tree Classifier",
                                                    "value":"tree"},
                                                    {"label":"K nearest neighbors model",
                                                    "value":"knn"},
                                                    {"label":"Best model", "value":"best"}],
                                            value = "best",
                                            placeholder="Select a model"),
                                html.Br(),
                                html.P(id="model-train"), 
                                html.P(id="model-test"),
                                html.Div(dcc.Graph(id="model-chart")) 
                               ])

# TASK 2:
# Add a callback function for `site-dropdown` as input, `success-pie-chart` as output
@app.callback(Output(component_id="success-pie-chart", component_property="figure"),
        Input(component_id="site-dropdown", component_property="value"))
def Pie(input_value):
    if input_value == "ALL":
        return px.pie(data_frame=spacex_df, names="Launch Site")
    else:
        mask = spacex_df["Launch Site"].isin([input_value])
        spacex_df2 = spacex_df[mask]
        return px.pie(data_frame=spacex_df2, names="class", labels={"1":"Succes", "2":"Failed"})
# TASK 4:
# Add a callback function for `site-dropdown` and `payload-slider` as inputs, `success-payload-scatter-chart` as output
@app.callback(Output(component_id="success-payload-scatter-chart", component_property="figure"),
              [Input(component_id="site-dropdown", component_property="value"),
              Input(component_id="payload-slider", component_property="value")])
def Scatter(input_value, payload):
    if input_value == "ALL":
        mask = (spacex_df["Payload Mass (kg)"] >= payload[0]) & (spacex_df["Payload Mass (kg)"] <= payload[1])
        spacex_df2 = spacex_df[mask]
        return px.scatter(x="Payload Mass (kg)", y="class", data_frame=spacex_df2, 
                          color="Booster Version Category")
    else:
        mask = spacex_df["Launch Site"].isin([input_value])
        spacex_df2 = spacex_df[mask]
        mask = (spacex_df2["Payload Mass (kg)"] >= payload[0]) & (spacex_df2["Payload Mass (kg)"] <= payload[1])
        spacex_df3 = spacex_df2[mask]
        return px.scatter(x="Payload Mass (kg)", y="class", data_frame=spacex_df3, 
                          color="Booster Version Category")
    


import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def plot_confusion_matrix(y,y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y, y_predict)
    return px.density_heatmap(cm, title="Confusion Matrix")
    
X = preprocessing.StandardScaler().fit_transform(pd.read_csv("dataset_part_3.csv"))
Y = pd.read_csv("dataset_part_2.csv").Class.to_numpy()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    
# Logistic regression
parameters_log ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge
lr=LogisticRegression()
logreg_cv = GridSearchCV(lr, param_grid=parameters_log, cv=10)
logreg_cv.fit(X_train, Y_train)
yhat_log = logreg_cv.predict(X_test)
    
# SVM
parameters_svm = {'kernel':('linear', 'rbf','poly','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()
svm_cv = GridSearchCV(svm, cv=10, param_grid=parameters_svm)
svm_cv.fit(X_train, Y_train)
yhat_svm = svm_cv.predict(X_test)
    
# Tree
parameters_tree = {'criterion': ['gini', 'entropy'],
                      'splitter': ['best', 'random'],
                      'max_depth': [2*n for n in range(1,10)],
                      'max_features': ['auto', 'sqrt'],
                      'min_samples_leaf': [1, 2, 4],
                      'min_samples_split': [2, 5, 10]}
tree = DecisionTreeClassifier()
tree_cv = GridSearchCV(tree, cv=10, param_grid = parameters_tree)
tree_cv.fit(X_train, Y_train)
yhat_tree = tree_cv.predict(X_test)

# KNN
parameters_knn = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                      'p': [1,2]}
KNN = KNeighborsClassifier()
knn_cv = GridSearchCV(KNN, cv=10, param_grid=parameters_knn)
knn_cv.fit(X_train, Y_train)
yhat_knn = knn_cv.predict(X_test)    

@app.callback(Output(component_id = "model-chart", component_property = "figure"),
              #Output(component_id = "model-train", component_property = "value"),
              #Output(component_id = "model-test", component_property = "value")],
             Input(component_id = "model-option", component_property = "value"))

def model_perform(input_value):
    print(input_value)
    if input_value == "logreg":
        return plot_confusion_matrix(Y_test,yhat_log)#, logreg_cv.best_score_, logreg_cv.score(X_test, Y_test)
    elif input_value == "svm":
        return plot_confusion_matrix(Y_test,yhat_svm)#, svm_cv.best_score_, svm_cv.score(X_test, Y_test)
    elif input_value == "tree":
        return plot_confusion_matrix(Y_test,yhat_tree)#, tree_cv.best_score_, tree_cv.score(X_test, Y_test)
    elif input_value == "knn":
        return plot_confusion_matrix(Y_test,yhat_knn)#, knn_cv.best_score_, knn_cv.score(X_test, Y_test)
    else:
        return px.bar(x= ["log", "svm", "tree", "knn"], 
            y=[logreg_cv.best_score_, svm_cv.best_score_, 
               tree_cv.best_score_, knn_cv.best_score_])

# Run the app
if __name__ == '__main__':
    app.run_server(host="0.0.0.0", debug=False)
