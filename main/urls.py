from django.conf.urls import url
from . import views
from django.views import generic

urlpatterns = [
    url(r'^$', views.IndexView.as_view(template_name='main/index.html'), name='index'),
    url(r'^all', views.AllView, name='all'),
    url(r'^pca', views.pcaView, name='pca'),
    url(r'^result_pca', views.pca_resultView, name='result_pca'),
    url(r'^feature', views.FeatureView, name='feature'),
    url(r'^Reduced', views.ReducedView, name='reduced'),
    url(r'^class', views.classView, name='class'),
    url(r'^show', views.showView, name='show'),
    url(r'^full_csv', views.full_csvView, name='full_csv'),
    url(r'^algos', views.algosView, name='algos'),
]