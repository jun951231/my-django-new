from django.conf.urls import url
from django.urls import path

from admin.common import views

urlpatterns = {
    path('', views.connection)
}