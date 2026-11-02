from django.contrib.auth.forms import AuthenticationForm
from django import forms


class BootstrapAuthenticationForm(AuthenticationForm):
    username = forms.CharField(widget=forms.TextInput(attrs={"class": "form-control", "autofocus": True}))
    password = forms.CharField(strip=False, widget=forms.PasswordInput(attrs={"class": "form-control"}))
