from django import template

register = template.Library()


@register.filter(name='add_class')
def add_class(field, css_class):
    """
    Rende il campo con classi CSS aggiuntive senza perdere attributi esistenti.
    """
    attrs = field.field.widget.attrs.copy()
    existing = attrs.get('class', '')
    attrs['class'] = f'{existing} {css_class}'.strip()
    return field.as_widget(attrs=attrs)
