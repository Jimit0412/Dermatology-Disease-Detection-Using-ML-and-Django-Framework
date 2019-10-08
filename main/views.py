from wsgiref.util import FileWrapper

from dask.dataframe.io import csv
from django.conf import settings
from django.views import generic
from django.http import HttpResponse

import pandas as pd
from algos import naive
from algos import bernoulliNB
from algos import knn
from algos import logreg
from algos import decisiontree
from algos import rf
from algos import svc
from algos import x


import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt, mpld3
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from django.shortcuts import render
from algos import pca
from matplotlib.pyplot import figure, title, bar, xlabel
import mpld3


class IndexView(generic.ListView):
    template_name = 'main/index.html'
    def get_queryset(self):
        return HttpResponse("abc")

def AllView(request):
    from matplotlib.pyplot import figure, title, bar
    df = pd.read_csv("dermatology_csv.csv")
    df = df.dropna()
    df_dummies = pd.get_dummies(df, columns=df.columns[0:-2])

    X = df_dummies.iloc[:, 2:].values

    y = df_dummies['class']

    na, ncm, navg = naive.gausnb(X, y)
    bna, bncm, bnavg = bernoulliNB.bernoulliNB(X, y)
    da, dcm, davg = decisiontree.decisiontree(X, y)
    ka, kcm, kavg = knn.knn(X, y)
    la, lcm, lavg = logreg.logreg(X, y)
    ra, rcm, ravg = rf.rf(X, y)
    sa, scm, savg = svc.svc(X, y)

    algos_list = [navg, bnavg, davg, kavg, lavg, ravg, savg]

    # fig, ax1 = plt.subplots()
    # fig.frameon = False

    # ax1.bar(range(len(algos_list)), [imp for imp in algos_list], align='center', color=('#44A744', '#D95F61'))
    # canvas = FigureCanvasAgg(fig)
    # response = HttpResponse(content_type='main/all_html')
    # canvas.print_png(response)
    # plt.close(fig)
    # return render(request, 'main/all_html.html', {'figure': response})

    mpl_figure = plt.figure(figsize=(10,5))
    algo_name = ['multiNB', 'bernoulliNB','decision tree','knn','logistic reg', 'random forest','svc']
    request.session['algo_name'] = algo_name
    plt.ylabel('Accuracy', fontsize = 20)

    #plt.title('Comparison of cross-validation average accuracy between various algorithms', fontsize= 30)
    ax = plt.bar(range(len(algos_list)), [imp for imp in algos_list], tick_label=algo_name, color=['red', 'lime', 'blue', 'cyan', 'magenta', 'yellow', 'grey'])
    rects = ax.patches

    for rect in rects:
        print(rect.get_height(), rect.get_y(), rect.get_x(), rect.get_width())
        x_value = rect.get_x()
        if(x_value<0):
            x_value=-0.74
        else:
            x_value-=0.45
        y_value = rect.get_height()/5
        print(rect.get_height(), rect.get_y(), rect.get_x(), rect.get_width(), y_value, x_value)
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar

        label = "{:.3f}".format(rect.get_height())
        plt.annotate(label,(x_value,y_value), xytext=(-17, space), textcoords="offset points", ha='right', va=va)
    # for v, i in enumerate(rects):
    #     plt.text(v, i, " "+str(v), va='center')
    fig_html = mpld3.fig_to_html(mpl_figure)
    title('Result')

    plt.close(mpl_figure)
    return render(request, 'main/all_html.html', {'figure': fig_html})


def pcaView(request):

    from matplotlib.pyplot import figure, title, bar
    df = pd.read_csv("dermatology_csv.csv")
    df = df.dropna()

    df_dummies = pd.get_dummies(df, columns=df.columns[0:-2])

    X = df_dummies.iloc[:, 2:].values
    mpl_figure = plt.figure(1, figsize=(6, 6))
    le, val = pca.pca_fun(X)
    plt.bar(range(le), [imp for imp in val])
    plt.xlabel('PCA Components')
    plt.ylabel('Variance')
    fig_html = mpld3.fig_to_html(mpl_figure)
    title('pca-components vs variance')
    plt.close(mpl_figure)
    return render(request, 'main/pca_html.html', {'figure': fig_html})
    #return render(request, 'music/pca_html.html', {"chart": type(z)})

def pca_resultView(request):

    df = pd.read_csv("dermatology_csv.csv")
    df = df.dropna()
    df_dummies = pd.get_dummies(df, columns=df.columns[0:-2])
    comp = int(request.GET.get('comp'))
    pcas = PCA(n_components=comp)
    X = pcas.fit_transform(df_dummies.iloc[:, 2:].values)

    y = df_dummies['class']

    na, ncm, navg = naive.gausnb(X, y)
    bna, bncm, bnavg = bernoulliNB.bernoulliNB(X, y)
    da, dcm, davg = decisiontree.decisiontree(X, y)
    ka, kcm, kavg = knn.knn(X, y)
    la, lcm, lavg = logreg.logreg(X, y)
    ra, rcm, ravg = rf.rf(X, y)
    sa, scm, savg = svc.svc(X, y)

    algos_list = [navg, bnavg, davg, kavg, lavg, ravg, savg]
    algo_name = ['multiNB', 'bernoulliNB', 'decision tree', 'knn', 'logistic reg', 'random forest', 'svc']
    mpl_figure = plt.figure(figsize=(8, 4))

    ax = plt.bar(range(len(algos_list)), [imp for imp in algos_list], tick_label=algo_name, color=['red', 'lime', 'blue', 'cyan', 'magenta', 'yellow', 'grey'])
    rects = ax.patches

    for rect in rects:
        print(rect.get_height(), rect.get_y(), rect.get_x(), rect.get_width())
        x_value = rect.get_x()
        if (x_value < 0):
            x_value = -0.74
        else:
            x_value -= 0.45
        y_value = rect.get_height() / 5
        print(rect.get_height(), rect.get_y(), rect.get_x(), rect.get_width(), y_value, x_value)
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar

        label = "{:.3f}".format(rect.get_height())
        plt.annotate(label, (x_value, y_value), xytext=(-17, space), textcoords="offset points", ha='right', va=va)
    fig_html = mpld3.fig_to_html(mpl_figure)
    title('REsult')
    plt.close(mpl_figure)
    return render(request, 'main/result_pca_html.html', {'figure': fig_html,'comp':comp})


def FeatureView(request):

    from sklearn.ensemble import ExtraTreesClassifier
    tree_clf = ExtraTreesClassifier(n_estimators=1500, criterion='entropy', random_state=24)
    df = pd.read_csv("dermatology_csv.csv")
    df = df.dropna()
    X = df.iloc[:, :34]
    y = df['class']

    tree_clf.fit(X, y)

    importances = tree_clf.feature_importances_
    feature_names = df.columns

    imp_features = dict(zip(feature_names, importances))
    features_up = sorted(imp_features.items(), key=lambda x: x[1], reverse=True)
    features_down = sorted(imp_features.items(), key=lambda x: x[1], reverse=False)



    features_only = [imp[0] for imp in features_up]
    request.session['features_down'] = features_down
    mpl_figure = plt.figure(figsize=(12, 6))

    plt.barh(range(len(features_up)), [imp[1] for imp in features_up])
    index = np.arange((len(features_only)))
    plt.yticks(index, features_only, rotation=30)
    plt.tight_layout()

    fig_html = mpld3.fig_to_html(mpl_figure)
    title('pca-components vs variance')
    plt.close(mpl_figure)
    algo_name = ['gausnb', 'bernoulliNB', 'decisiontree', 'knn', 'logreg', 'rf', 'svc']
    return render(request, 'main/feature_imp.html', {'figure': fig_html,'algo_name':algo_name})

    # plt.bar(range(len(features_up)), [imp[1] for imp in features_up], align='center')
    # plt.title('features');
    #
    # final_features = []
    # for f in range(24):
    #     final_features.append(features_down[f][0])
    #
    # df_small2 = df.drop(labels=final_features, axis=1)
    # print("Current shape of dataset :", df_small2.shape)

def ReducedView(request):

    final_features = []
    features_down = request.session['features_down']
    for f in range(24):
        final_features.append(features_down[f][0])

    df = pd.read_csv("dermatology_csv.csv")
    df = df.dropna()
    df_small2 = df.drop(labels=final_features, axis=1)

    ds = pd.get_dummies(df_small2, columns=df_small2.columns[0:-1])

    X = ds.iloc[:, 1:].values
    y = df['class']

    na, ncm, navg = naive.gausnb(X, y)
    bna, bncm, bnavg = bernoulliNB.bernoulliNB(X, y)
    da, dcm, davg = decisiontree.decisiontree(X, y)
    ka, kcm, kavg = knn.knn(X, y)
    la, lcm, lavg = logreg.logreg(X, y)
    ra, rcm, ravg = rf.rf(X, y)
    sa, scm, savg = svc.svc(X, y)


    algos_list = [navg, bnavg, davg, kavg, lavg, ravg, savg]
    algo_name = ['multiNB', 'bernoulliNB', 'decision tree', 'knn', 'logistic reg', 'random forest', 'svc']
    mpl_figure = plt.figure(figsize=(8, 4))

    ax = plt.bar(range(len(algos_list)), [imp for imp in algos_list], tick_label=algo_name, color=['red', 'lime', 'blue', 'cyan', 'magenta', 'yellow', 'grey'])
    rects = ax.patches

    for rect in rects:
        print(rect.get_height(), rect.get_y(), rect.get_x(), rect.get_width())
        x_value = rect.get_x()
        if (x_value < 0):
            x_value = -0.74
        else:
            x_value -= 0.45
        y_value = rect.get_height() / 5
        print(rect.get_height(), rect.get_y(), rect.get_x(), rect.get_width(), y_value, x_value)
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar

        label = "{:.3f}".format(rect.get_height())
        plt.annotate(label, (x_value, y_value), xytext=(-17, space), textcoords="offset points", ha='right', va=va)
    fig_html = mpld3.fig_to_html(mpl_figure)
    title('REsult')
    plt.close(mpl_figure)
    return render(request, 'main/reduced_html.html', {'figure': fig_html})


def algosView(request):
    navgl = []
    algo = request.GET.get('algo')
    for i in range(1,33):
        final_features = []
        features_down = request.session['features_down']
        for f in range(i):
            final_features.append(features_down[f][0])

        df = pd.read_csv("dermatology_csv.csv")
        df = df.dropna()
        df_small2 = df.drop(labels=final_features, axis=1)

        ds = pd.get_dummies(df_small2, columns=df_small2.columns[0:-1])

        X = ds.iloc[:, 1:].values
        y = df['class']
        navgl.append(getattr(x, algo)(X, y))

    mpl_figure = plt.figure(figsize=(8, 4))
    plt.bar(range(len(navgl)), [imp for imp in navgl], tick_label=range(1,33)[::-1])
    plt.ylabel('Accuracy', fontsize = 20)
    plt.xlabel('Number of features trained',fontsize = 20)
    plt.yticks(np.arange(0, 1.05, step=0.05))
    fig_html = mpld3.fig_to_html(mpl_figure)
    title('REsult')
    plt.close(mpl_figure)
    return render(request, 'main/algo.html', {'figure': fig_html, 'algo':algo})


def classView(request):
    template_name = 'main/class_html.html'
    return render(request, 'main/class.html')

def showView(request):

    data = request.GET.get('num')
    final_features = []
    features_down = request.session['features_down']
    for f in range(24):
        final_features.append(features_down[f][0])
    df = pd.read_csv("dermatology_csv.csv")
    df = df.dropna()
    df_small2 = df.drop(labels=final_features, axis=1)

    ds = pd.get_dummies(df_small2, columns=df_small2.columns[0:-1])
    new_df = ds.loc[ds['class'] == int(data)]

    X = ds.iloc[:, 1:].values
    y = ds['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=24)

    X_test = new_df.iloc[:, 1:].values
    y_test = new_df['class']

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    gcm = confusion_matrix(y_test, y_pred)
    navg = accuracy_score(y_test, y_pred)


    sv = svm.SVC()
    sv.fit(X_train, y_train)

    y_pred = sv.predict(X_test)

    scm = confusion_matrix(y_test, y_pred)

    savg = accuracy_score(y_test, y_pred)

    clf = LogisticRegression(random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    lcm = confusion_matrix(y_test, y_pred)

    lavg = accuracy_score(y_test, y_pred)




    bnb = BernoulliNB()
    bnb.fit(X_train, y_train)
    y_pred = bnb.predict(X_test)

    bcm = confusion_matrix(y_test, y_pred)

    bnavg = accuracy_score(y_test, y_pred)

    tree = DecisionTreeClassifier()
    tree.fit(X_train, y_train)

    y_pred = tree.predict(X_test)

    dcm = confusion_matrix(y_test, y_pred)
    davg = accuracy_score(y_test, y_pred)


    knn_clf = KNeighborsClassifier()
    knn_clf.fit(X_train, y_train)
    y_pred = knn_clf.predict(X_test)

    kcm = confusion_matrix(y_test, y_pred)
    kavg = accuracy_score(y_test, y_pred)


    rf = RandomForestClassifier(n_estimators=5)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    rcm = confusion_matrix(y_test, y_pred)
    ravg = accuracy_score(y_test, y_pred)


    algos_list = [navg, bnavg, davg, kavg, lavg, ravg, savg]

    algo_name = ['GaussianNB', 'bernoulliNB', 'decision tree', 'knn', 'logistic reg', 'random forest', 'svc']

    cm_list = (gcm, bcm, dcm, kcm, lcm, rcm, scm)

    cm_name = ['GaussianNB', 'bernoulliNB', 'decision tree', 'knn', 'logistic reg', 'random forest', 'svc']

    c = {e: cm_list[i] for i, e in enumerate(cm_name)}
   # dic = dict(zip(cm_list, cm_name))
    mpl_figure = plt.figure(figsize=(8, 4))

    ax = plt.bar(range(len(algos_list)), [imp for imp in algos_list], tick_label=algo_name, color=['red', 'lime', 'blue', 'cyan', 'magenta', 'yellow', 'grey'])
    rects = ax.patches

    for rect in rects:
        print(rect.get_height(), rect.get_y(), rect.get_x(), rect.get_width())
        x_value = rect.get_x()
        if (x_value < 0):
            x_value = -0.74
        else:
            x_value -= 0.45
        y_value = rect.get_height() / 5
        print(rect.get_height(), rect.get_y(), rect.get_x(), rect.get_width(), y_value, x_value)
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar

        label = "{:.3f}".format(rect.get_height())
        plt.annotate(label, (x_value, y_value), xytext=(-17, space), textcoords="offset points", ha='right', va=va)
    fig_html = mpld3.fig_to_html(mpl_figure)
    title('REsult')
    plt.close(mpl_figure)
    return render(request, 'main/show.html', {'figure': fig_html, 'confusion': c})

def full_csvView(request):
    import io
    #
    # # buffer = io.StringIO()
    # # wr = csv.writer(buffer, quoting=csv.QUOTE_ALL)
    # # wr.writerows(file_rows)
    # #
    # # buffer.seek(0)
    # response = HttpResponse(content_type='text/csv')
    # response['Content-Disposition'] = 'attachment; filename=dermatology_csv.csv'
    import csv
    # with open('dermatology_csv.csv', 'wb') as myfile:  # python 3: open('stockitems_misuper.csv', 'w', newline="")
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #
    #
    #     wr.writerows()
    #
    # with open('dermatology_csv.csv', 'rb') as myfile:
    #     response = HttpResponse(myfile, content_type='text/csv')
    #     response['Content-Disposition'] = 'attachment; filename=dermatology_csv.csv'
    #     return response
    filename = '/home/jimit/Desktop/my_website/main/dermatology_csv.csv'
    # filename= r"C:\Users\A6B0SZZ\PycharmProjects\sample\media\output1.csv"
    download_name = "dermatology_csv.csv"
    wrapper = FileWrapper(open(filename))
    response = HttpResponse(wrapper, content_type='text/csv')
    response['Content-Disposition'] = "attachment; filename=%s" % download_name
    return response
