{{ name | escape | underline}}

.. currentmodule:: {{ module }}

{{ fullname }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:

   {% block methods %}
   {% block attributes %}
   {% if attributes %}
      .. autosummary::
         :toctree:
      {% for item in all_attributes %}
         {%- if not item.startswith('_') %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
   {% endif %}
   {% endblock %}

   {% if methods %}
      .. autosummary::
         :toctree:
      {% for item in all_methods %}
         {%- if not item.startswith('_') or item in ['__call__'] %}
         {{ name }}.{{ item }}
         {%- endif -%}
      {%- endfor %}
   {% endif %}
   {% endblock %}
