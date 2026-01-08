# ui/urls.py
from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('login/', views.user_login, name='login'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
    path('register/', views.user_register, name='register'),
    path('', views.index, name='home'),
    path('upload/', views.upload, name='upload'),
    path('chat/', views.chat, name='chat'),
    path('public-chat/', views.public_chat, name='public_chat'),
    path('password/change/', views.change_password, name='change_password'),
    path('uploads/', views.list_uploads, name='list_uploads'),
]
