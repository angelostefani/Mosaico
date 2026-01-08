# ui/forms.py
from django import forms

class UploadForm(forms.Form):
    file = forms.FileField()
    username = forms.CharField(required=False)
    collection = forms.CharField(required=False)

class ChatForm(forms.Form):
    question = forms.CharField(widget=forms.TextInput(attrs={'size':60}))
    username = forms.CharField(required=False)
    collection = forms.CharField(required=False)
