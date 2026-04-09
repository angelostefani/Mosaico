# ui/urls.py
from django.urls import path
from . import views
from django.contrib.auth.views import LogoutView

urlpatterns = [
    path('set-language/', views.set_language, name='set_language'),
    path('login/', views.user_login, name='login'),
    path('logout/', LogoutView.as_view(next_page='home'), name='logout'),
    path('register/', views.user_register, name='register'),
    path('', views.index, name='home'),
    path('upload/', views.upload, name='upload'),
    path('chat/', views.chat, name='chat'),
    path('collection-config/', views.collection_config, name='collection_config'),
    path('public-chat/', views.public_chat, name='public_chat'),
    path('password/change/', views.change_password, name='change_password'),
    path('uploads/', views.list_uploads, name='list_uploads'),
    path('conversations/<str:conversation_id>/delete/', views.delete_conversation, name='delete_conversation'),
    path('uploads/<str:upload_id>/delete/', views.delete_upload, name='delete_upload'),
]
