from .i18n import TRANSLATIONS, SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE


def i18n(request):
    """
    Aggiunge al contesto di ogni template:
    - ``lang``: codice lingua corrente ('it' o 'en')
    - ``t``: dizionario delle traduzioni per la lingua corrente
    """
    lang = request.session.get('_language', DEFAULT_LANGUAGE)
    if lang not in SUPPORTED_LANGUAGES:
        lang = DEFAULT_LANGUAGE
    return {
        'lang': lang,
        't': TRANSLATIONS[lang],
    }
