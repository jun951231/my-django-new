from django.conf.urls import url

from admin.user import views

urlpatterns = {
    url(r'^join', views.users, name='users'),
    url(r'^list', views.users, name='users'),
    url(r'^modify', views.users, name='users'),
    url(r'^login', views.login, name='login'),
    url(r'^delete/<slug:id>', views.users),
    url(r'^detail/(?P<username>\w{0,50})/$', views.detail),
}