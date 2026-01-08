# ui/views.py
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
import json
import requests
from .forms import UploadForm, ChatForm
from django.contrib.auth import authenticate, login
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.forms import PasswordChangeForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib.auth import update_session_auth_hash

API_BASE = getattr(settings, 'API_BASE', 'http://localhost:9000')
DJANGO_TOKEN_URL = '/api/token/'
FAKE_TOKEN = getattr(settings, 'FAKE_TOKEN', 'dev-token')

def _obtain_jwt(request, username, password):
    """
    Helper that asks the backend for a JWT and stores it in session.
    Returns True on success so callers can decide how to proceed otherwise.
    """
    try:
        resp = requests.post(
            request.build_absolute_uri(DJANGO_TOKEN_URL),
            json={'username': username, 'password': password},
            timeout=10,
        )
    except requests.RequestException:
        return False
    if resp.status_code == 200:
        token = resp.json().get('access')
        if token:
            request.session['jwt_token'] = token
            return True
    return False

def user_login(request):
    token = None
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            # login Django session
            user = form.get_user()
            login(request, user)

            # ottieni JWT da DRF SimpleJWT
            if _obtain_jwt(request, form.cleaned_data['username'], form.cleaned_data['password']):
                return redirect('home')
            form.add_error(None, 'Impossibile ottenere JWT')
    else:
        form = AuthenticationForm()
    return render(request, 'ui/login.html', {'form': form, 'token': token})

def user_register(request):
    """
    Mostra il form di registrazione e crea un nuovo utente.
    Al termine, esegue il login automatico e reindirizza alla home.
    """
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()          # crea l'utente
            login(request, user)        # effettua subito il login
            username = form.cleaned_data['username']
            password = form.cleaned_data['password1']
            if not _obtain_jwt(request, username, password):
                # Manteniamo l'utente autenticato ma segnaliamo la mancanza del token
                request.session['jwt_token_error'] = 'Impossibile ottenere JWT, alcune funzioni potrebbero non funzionare.'
            return redirect('home')
    else:
        form = UserCreationForm()
    return render(request, 'ui/register.html', {'form': form})

@login_required
def change_password(request):
    """
    Allow authenticated users to update their password keeping the session valid.
    """
    if request.method == 'POST':
        form = PasswordChangeForm(request.user, request.POST)
        if form.is_valid():
            user = form.save()
            update_session_auth_hash(request, user)
            messages.success(request, 'Password aggiornata correttamente.')
            return redirect('home')
        messages.error(request, 'Correggi gli errori indicati e riprova.')
    else:
        form = PasswordChangeForm(request.user)
    return render(request, 'ui/change_password.html', {'form': form})

@login_required
def index(request):
    return render(request, 'ui/index.html')

@login_required
def upload(request):
    status = None
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            files = {'file': request.FILES['file']}
            data = {
                'username': form.cleaned_data['username'],
                'collection': form.cleaned_data['collection'],
            }
            headers = {'Authorization': f'Bearer {FAKE_TOKEN}'}
            r = requests.post(f'{API_BASE}/upload', files=files, data=data, headers=headers)
            status = (r.status_code, r.json() if r.ok else r.text)
    else:
        form = UploadForm()
    warning = request.session.pop('jwt_token_error', None)
    context = {
        'form': form,
        'status': status,
        'jwt_warning': warning,
        'fake_token': FAKE_TOKEN,
    }
    return render(request, 'ui/upload.html', context)

@login_required
def chat(request):
    response = None
    if request.method == 'POST':
        form = ChatForm(request.POST)
        if form.is_valid():
            data = {
                'question': form.cleaned_data['question'],
                'username': form.cleaned_data['username'],
                'collection': form.cleaned_data['collection'],
            }
            headers = {'Authorization': f'Bearer {FAKE_TOKEN}'}
            r = requests.post(f'{API_BASE}/chat', data=data, headers=headers)
            if r.ok:
                response = r.json().get('message')
            else:
                response = f'Errore: {r.status_code}'
    else:
        form = ChatForm()
    return render(request, 'ui/chat.html', {'form': form, 'response': response})

@login_required
def list_uploads(request):
    uploads = None
    error = None
    if request.GET:
        params = {
            'username': request.GET.get('username'),
            'collection': request.GET.get('collection'),
            'limit': request.GET.get('limit', 100)
        }
        headers = {'Authorization': f'Bearer {FAKE_TOKEN}'}
        r = requests.get(f'{API_BASE}/uploads', params=params, headers=headers)
        if r.ok:
            uploads = r.json()
        else:
            error = r.text
    context = {
        'uploads': uploads,
        'error': error,
        'api_base': settings.API_BASE,
        'fake_token': FAKE_TOKEN,
    }
    return render(request, 'ui/uploads.html', context)

def public_chat(request):
    """
    Chat page without authentication that relies on username and collection
    supplied via query string. Values are not editable client-side.
    """
    username = (request.GET.get('username') or '').strip()
    collection = (request.GET.get('collection') or '').strip()
    status_session_key = f'public_chat_status::{username}::{collection}'
    error_session_key = f'public_chat_error::{username}::{collection}'

    stored_error = request.session.get(error_session_key)
    stored_status = request.session.get(status_session_key)

    context = {
        'username': username,
        'collection': collection,
        'api_base': API_BASE,
        'question': '',
        'history': [],
        'error': stored_error,
        'status_code': stored_status,
    }

    if not username or not collection:
        context['error'] = 'Parametri "username" e "collection" obbligatori nella querystring.'
        return render(request, 'ui/public_chat.html', context, status=400)

    session_key = f'public_chat_history::{username}::{collection}'
    history = request.session.get(session_key, [])

    if request.method == 'POST' and request.content_type == 'application/json':
        try:
            payload = json.loads(request.body.decode('utf-8') or '{}')
        except ValueError:
            return JsonResponse({'error': 'Payload non valido.'}, status=400)

        question = (payload.get('question') or '').strip()
        answer = payload.get('answer')
        error = payload.get('error')
        status_code = payload.get('status_code')

        if question:
            history.append({'role': 'user', 'message': question})

        if answer:
            if not isinstance(answer, str):
                answer = str(answer)
            history.append({'role': 'assistant', 'message': answer})

        if error:
            context['error'] = error
            request.session[error_session_key] = error
        else:
            request.session.pop(error_session_key, None)

        if status_code is not None:
            context['status_code'] = status_code
            request.session[status_session_key] = status_code
        else:
            request.session.pop(status_session_key, None)

        request.session[session_key] = history
        request.session.modified = True

        context['history'] = history

        return JsonResponse(
            {
                'history': history,
                'error': context['error'],
                'status_code': context['status_code'],
            }
        )

    elif request.method == 'POST':
        # Fallback non-JS: inform the user that the feature requires JavaScript.
        question = (request.POST.get('question') or '').strip()
        context['question'] = question
        error_message = (
            'L\'invio pubblico richiede JavaScript abilitato per contattare il servizio di chat.'
        )
        context['error'] = error_message
        request.session[error_session_key] = error_message
    else:
        context['question'] = ''

    context['history'] = history
    context['fake_token'] = FAKE_TOKEN

    if request.method != 'POST':
        # Clear any previous state so the banner does not linger on next load.
        request.session.pop(error_session_key, None)
        request.session.pop(status_session_key, None)

    return render(request, 'ui/public_chat.html', context)

# Create your views here.
