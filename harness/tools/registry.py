"""
registry.py — Registro de herramientas del agente.

Este módulo implementa el mecanismo central que permite al agente usar
herramientas (tools). Sus tres responsabilidades son:

  1. **Almacenar** las herramientas registradas, indexadas por nombre.
  2. **Generar esquemas JSON** compatibles con el parámetro ``tools=`` de
     la API de OpenAI, para que el modelo sepa qué funciones existen,
     qué hacen y qué argumentos esperan.
  3. **Despachar llamadas**: cuando el modelo responde con una solicitud
     de ejecución de herramienta, este módulo localiza la función
     correspondiente, la ejecuta con los argumentos proporcionados y
     devuelve el resultado como texto.

Componentes principales:

  - ``Tool``          → Dataclass que agrupa el nombre, la descripción,
                        la función Python y el esquema JSON de una herramienta.
  - ``ToolRegistry``  → Contenedor que gestiona la colección de herramientas,
                        genera sus esquemas y ejecuta las llamadas entrantes.
  - ``registry``      → Instancia global y única del registro. Es el punto
                        de acceso que importa el resto del harness.
  - ``@tool``         → Decorador que registra automáticamente una función
                        como herramienta al definir el módulo que la contiene.

Flujo típico:

  1. Un módulo de herramientas (p. ej. ``filesystem.py``) define funciones
     decoradas con ``@tool``.
  2. Al importarse ese módulo, el decorador extrae metadatos (nombre,
     docstring, esquema de parámetros) y los registra en ``registry``.
  3. El agente llama a ``registry.get_schemas()`` para obtener la lista
     de herramientas y enviarla al modelo en cada petición.
  4. Cuando el modelo responde con ``tool_calls``, el agente llama a
     ``registry.dispatch(nombre, argumentos)`` para ejecutar la función.

Uso:
    from harness.tools.registry import registry, tool

    @tool
    def mi_herramienta(param: str) -> str:
        '''Descripción de la herramienta.'''
        return f"resultado: {param}"
"""

# ============================================================
# IMPORTACIONES
# ============================================================

# inspect: módulo estándar para introspección de objetos Python.
# Aquí se usa para extraer el docstring de una función de forma limpia
# (inspect.getdoc() elimina la indentación heredada del código fuente,
# a diferencia de acceder directamente a func.__doc__).
import inspect

# Callable: tipo genérico del módulo collections.abc que representa
# cualquier objeto invocable (funciones, métodos, lambdas, clases, etc.).
# Se usa en las anotaciones de tipo para indicar "esto es una función".
from collections.abc import Callable

# dataclass: decorador del módulo estándar que genera automáticamente
# los métodos especiales (__init__, __repr__, __eq__, etc.) de una clase
# a partir de las anotaciones de tipo de sus atributos. Elimina el
# boilerplate de escribir constructores manualmente.
from dataclasses import dataclass

# Any: tipo comodín de typing que acepta cualquier valor. Se usa en el
# diccionario de argumentos del despacho porque los argumentos varían
# según la herramienta (pueden ser str, int, bool, etc.).
from typing import Any

# TypeAdapter: clase de Pydantic v2 que permite generar esquemas JSON
# a partir de tipos Python arbitrarios (incluidas firmas de funciones).
# Es la pieza clave que convierte las anotaciones de tipo de una función
# Python en el formato JSON Schema que espera la API de OpenAI.
from pydantic import TypeAdapter

# ============================================================
# DATACLASS: Tool
# ============================================================
# Representa una herramienta registrada. Agrupa toda la información
# necesaria para exponer una función Python como herramienta del agente.
#
# Se usa @dataclass para que Python genere automáticamente el constructor
# (__init__) y otros métodos a partir de los atributos declarados.
# Esto evita escribir código repetitivo de inicialización.

@dataclass
class Tool:
    """
    Representa una herramienta registrada.

    Agrupa toda la información necesaria para exponer una función Python
    como herramienta del agente: su identidad (nombre), su documentación
    (descripción), la referencia a la función ejecutable y el esquema JSON
    de sus parámetros.

    Atributos:
        name:        Nombre único de la herramienta. Es el identificador que
                     el modelo usa en sus llamadas (``tool_calls[].function.name``).
                     Coincide con el nombre de la función Python decorada.
        description: Descripción textual de lo que hace la herramienta. Se
                     extrae del docstring de la función y se envía al modelo
                     para que sepa cuándo y cómo usarla.
        function:    Referencia a la función Python ejecutable. Se invoca con
                     ``function(**arguments)`` durante el despacho.
        schema:      Esquema JSON de los parámetros de la función, en formato
                     compatible con OpenAI. Se genera automáticamente a partir
                     de las anotaciones de tipo de la función mediante Pydantic.
    """
    # Nombre único de la herramienta. Ejemplo: "read", "git_commit", "bash".
    # El modelo usa este nombre exacto al solicitar una herramienta.
    name: str

    # Descripción legible de la herramienta. Proviene del docstring.
    # El modelo la lee para decidir cuándo usar la herramienta.
    description: str

    # Referencia directa a la función Python. No es una cadena ni un nombre:
    # es el objeto función real, invocable con function(**kwargs).
    function: Callable

    # Esquema JSON de los parámetros, generado por Pydantic.
    # Describe los nombres, tipos y obligatoriedad de cada argumento.
    # Ejemplo para `def read(path: str) -> str:`:
    # {
    #   "type": "object",
    #   "properties": {"path": {"type": "string"}},
    #   "required": ["path"]
    # }
    schema: dict


# ============================================================
# CLASE: ToolRegistry
# ============================================================
# Registro central de herramientas. Implementa el patrón de diseño
# "Registry" (Registro): un punto único de acceso para almacenar y
# recuperar objetos por nombre.
#
# Responsabilidades:
#   - register():    añadir una herramienta al registro.
#   - get_schemas(): generar la lista de esquemas para enviar al modelo.
#   - dispatch():    ejecutar una herramienta por nombre con argumentos.
#
# Existe una única instancia global (variable `registry` al final del
# módulo) que se comparte entre todos los módulos del harness.

class ToolRegistry:
    """
    Registro central de herramientas.

    Almacena las herramientas registradas, expone sus esquemas en el formato
    que espera la API de OpenAI y despacha las llamadas entrantes del modelo.
    Existe una única instancia global (``registry``) que se comparte entre
    todos los módulos del harness.
    """

    def __init__(self) -> None:
        """
        Inicializa el registro con una colección vacía de herramientas.

        Se usa un diccionario indexado por nombre para que la búsqueda
        durante el despacho sea O(1) (acceso directo por hash) en lugar
        de O(n) (búsqueda lineal en una lista).
        """
        # Diccionario interno que mapea:
        #   clave   → nombre de la herramienta (str)
        #   valor   → objeto Tool con metadatos + función
        #
        # El prefijo "_" indica que es un atributo "privado por convención":
        # el acceso externo debe hacerse a través de los métodos públicos
        # register(), get_schemas() y dispatch(). Python no impone
        # privacidad real, pero es una señal de intención.
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Registra una herramienta en el registro.

        Si ya existe una herramienta con el mismo nombre, se sobrescribe
        sin aviso (política «última escritura gana»). Esto permite, por
        ejemplo, redefinir una herramienta durante el desarrollo sin tener
        que reiniciar el proceso.

        Args:
            tool: Objeto Tool con los metadatos y la función a registrar.
        """
        # Inserción o sobrescritura en el diccionario.
        # Si el nombre ya existía, la nueva definición reemplaza a la anterior.
        # No se emite ningún aviso ni excepción: es una decisión de diseño
        # para facilitar la iteración rápida durante el desarrollo.
        self._tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        """
        Devuelve la lista de esquemas de herramientas en el formato OpenAI.

        La API de chat completions de OpenAI espera que el parámetro
        ``tools=`` sea una lista de objetos con la estructura::

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "parameters": { ... }  # esquema JSON Schema
                }
            }

        Este método construye esa estructura para cada herramienta
        registrada y la devuelve como lista. Se llama en cada petición
        al modelo para que conozca las herramientas disponibles.

        Returns:
            Lista de diccionarios, uno por herramienta registrada, en el
            formato que espera ``client.chat.completions.create(tools=...)``.

        Ejemplo de salida con una herramienta "read":
            [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Lee un fichero del workspace.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"}
                            },
                            "required": ["path"]
                        }
                    }
                }
            ]
        """
        # Comprensión de lista que itera sobre todos los objetos Tool
        # registrados y los envuelve en la estructura que exige la API
        # de OpenAI. El campo "parameters" contiene el esquema JSON
        # generado por Pydantic, que describe los tipos, nombres y
        # obligatoriedad de cada parámetro de la función.
        return [
            {
                "type": "function",       # Tipo fijo: siempre "function"
                "function": {
                    "name": t.name,              # Identificador de la herramienta
                    "description": t.description, # Docstring de la función
                    "parameters": t.schema,       # Esquema JSON Schema
                },
            }
            for t in self._tools.values()  # Iterar sobre todos los Tool registrados
        ]

    def dispatch(self, name: str, arguments: dict[str, Any]) -> str:
        """
        Ejecuta la herramienta indicada con los argumentos dados.

        Este es el punto de ejecución real: el agente recibe una solicitud
        de herramienta del modelo (nombre + argumentos JSON), la pasa aquí,
        y este método localiza la función, la invoca y devuelve el resultado
        como cadena de texto.

        El resultado siempre es un string porque es lo que se inserta en el
        mensaje de rol ``"tool"`` del historial de conversación. Si la
        función devuelve un tipo distinto, se convierte con ``str()``.

        Las excepciones se capturan internamente y se devuelven como strings
        de error. Esto evita que un fallo en una herramienta detenga el bucle
        del agente: el modelo ve el mensaje de error y decide cómo proceder
        (reintentar, corregir argumentos, informar al usuario, etc.).

        Args:
            name:      Nombre de la herramienta a ejecutar. Debe coincidir
                       con el nombre registrado (normalmente el nombre de la
                       función Python decorada con @tool).
            arguments: Diccionario de argumentos que se pasan a la función
                       como keyword arguments (**arguments). Proviene de la
                       deserialización JSON de ``tool_calls[].function.arguments``.

        Returns:
            Resultado de la herramienta como cadena, o un mensaje de error
            si el nombre es desconocido o la función lanza una excepción.

        Ejemplo de uso:
            resultado = registry.dispatch("read", {"path": "src/main.py"})
            # resultado → "import os\\nimport sys\\n..."
        """
        # Validación previa: rechazar nombres de herramienta desconocidos.
        # Devolvemos un string de error (en lugar de lanzar una excepción)
        # para que el modelo pueda leerlo y recuperarse: por ejemplo,
        # corrigiendo el nombre o informando al usuario de que la
        # herramienta no existe.
        if name not in self._tools:
            return f"error: herramienta desconocida '{name}'"

        # Ejecutar la herramienta dentro de un bloque try/except para que
        # cualquier excepción (FileNotFoundError, PermissionError, ValueError,
        # TypeError, etc.) no propague hacia arriba y rompa el bucle del agente.
        # El modelo recibe el string de error y decide el siguiente paso.
        try:
            # Desempaquetamos el diccionario de argumentos como keyword
            # arguments de la función usando el operador **.
            # Por ejemplo, si arguments es {"path": "src/main.py"},
            # la llamada equivale a: function(path="src/main.py")
            result = self._tools[name].function(**arguments)

            # Convertimos el resultado a string para que pueda insertarse
            # en el mensaje de rol "tool" del historial de conversación.
            # Las funciones de herramientas ya devuelven strings por diseño,
            # pero esta conversión es una red de seguridad por si alguna
            # devuelve otro tipo (int, bool, None, etc.).
            return str(result)

        except Exception as e:
            # Capturamos CUALQUIER excepción (Exception es la clase base
            # de todas las excepciones no de sistema) y la formateamos como
            # un string legible que incluye el tipo de excepción y su mensaje.
            #
            # Ejemplo de salida:
            #   "error: FileNotFoundError: [Errno 2] No such file or directory: 'foo.py'"
            #   "error: PermissionError: [Errno 13] Permission denied: '/etc/passwd'"
            #
            # El modelo lee este string, comprende que algo falló, y puede
            # decidir reintentar con argumentos corregidos, usar otra
            # herramienta, o informar al usuario del problema.
            return f"error: {type(e).__name__}: {e}"


# ============================================================
# INSTANCIA GLOBAL DEL REGISTRO
# ============================================================
# Este es el único registro que usa todo el harness. Se crea una sola vez,
# en el momento de importar este módulo, y se comparte entre todos los
# demás módulos.
#
# - Los módulos de herramientas (filesystem.py, git.py, bash.py, etc.)
#   importan `tool` (el decorador) para registrar sus funciones.
# - El bucle principal del agente importa `registry` para obtener los
#   esquemas (get_schemas) y ejecutar herramientas (dispatch).
#
# Es un patrón de "singleton por módulo": no se usa una clase Singleton,
# simplemente se crea una instancia a nivel de módulo y se importa esa.
registry = ToolRegistry()


# ============================================================
# DECORADOR: @tool
# ============================================================
# Este decorador es el mecanismo de registro automático. Cuando un módulo
# de herramientas se importa, todas las funciones decoradas con @tool se
# registran inmediatamente en la instancia global `registry`.
#
# El decorador NO envuelve la función (no crea un wrapper). Solo extrae
# metadatos, registra la herramienta y devuelve la función original intacta.
# Esto significa que la función puede seguir usándose como una función
# Python normal en tests u otros contextos.
#
# El momento de ejecución del decorador es el import del módulo:
#   import harness.tools.filesystem  →  @tool se ejecuta para cada función
#   import harness.tools.git         →  @tool se ejecuta para cada función
#   ...
# Al final de todos los imports, `registry` contiene todas las herramientas.

def tool(func: Callable) -> Callable:
    """
    Decorador que registra una función como herramienta del agente.

    Al aplicarse sobre una función, este decorador:

      1. Extrae el nombre de la función (``func.__name__``).
      2. Extrae el docstring como descripción para el modelo.
      3. Genera un esquema JSON de los parámetros usando Pydantic, a partir
         de las anotaciones de tipo de la función.
      4. Registra la herramienta en el registro global ``registry``.
      5. Devuelve la función original sin modificar, para que pueda seguir
         llamándose directamente desde código Python si es necesario.

    El decorador se ejecuta en el momento del import del módulo que lo
    contiene. Por eso los módulos de herramientas se importan por sus
    efectos secundarios en ``harness/tools/__init__.py``.

    Args:
        func: La función que se quiere registrar como herramienta. Debe
              tener anotaciones de tipo en sus parámetros para que Pydantic
              pueda generar el esquema JSON correctamente.

    Returns:
        La misma función recibida, sin envoltorio ni modificación. El
        registro es un efecto secundario del decorador, no transforma
        la función.

    Ejemplo::

        @tool
        def read(path: str) -> str:
            '''Lee un fichero del workspace.'''
            return Path(path).read_text()

        # Tras el import, registry contiene una herramienta llamada "read"
        # con el docstring como descripción y el siguiente esquema:
        # {
        #   "type": "object",
        #   "properties": {"path": {"type": "string"}},
        #   "required": ["path"]
        # }
    """
    # ----------------------------------------------------------
    # Paso 1: Extraer metadatos de la función.
    # ----------------------------------------------------------

    # El nombre de la función se usa como identificador único en el registro
    # y como el nombre que el modelo verá en la lista de herramientas.
    # Ejemplo: def read(...) → name = "read"
    name = func.__name__

    # El docstring se usa como descripción para el modelo.
    # inspect.getdoc() es preferible a func.__doc__ porque:
    #   - Limpia la indentación heredada del código fuente.
    #   - Devuelve None si no hay docstring (en vez de un string vacío).
    # Usamos `or ""` para convertir None en cadena vacía si no hay docstring.
    description = inspect.getdoc(func) or ""

    # ----------------------------------------------------------
    # Paso 2: Generar el esquema JSON de los parámetros.
    # ----------------------------------------------------------
    # TypeAdapter de Pydantic introspecciona la firma de la función
    # (nombres de parámetros, anotaciones de tipo, valores por defecto)
    # y produce un esquema JSON Schema compatible con OpenAI.
    #
    # Ejemplo: para `def read(path: str) -> str:` genera:
    #   {
    #     "type": "object",
    #     "properties": {
    #       "path": {"type": "string"}
    #     },
    #     "required": ["path"]
    #   }
    #
    # Ejemplo: para `def git_log(limit: int = 10) -> str:` genera:
    #   {
    #     "type": "object",
    #     "properties": {
    #       "limit": {"type": "integer", "default": 10}
    #     }
    #   }
    # (sin "required" porque limit tiene valor por defecto)
    #
    # IMPORTANTE: la función DEBE tener anotaciones de tipo en sus
    # parámetros. Sin ellas, Pydantic no puede generar el esquema.
    schema = TypeAdapter(func).json_schema()

    # ----------------------------------------------------------
    # Paso 3: Crear el objeto Tool y registrarlo.
    # ----------------------------------------------------------
    # Se construye un objeto Tool con todos los metadatos extraídos
    # y se registra en la instancia global del registro. A partir de
    # este momento, la herramienta:
    #   - Aparece en registry.get_schemas() (el modelo puede verla).
    #   - Puede ser despachada por nombre con registry.dispatch().
    registry.register(Tool(
        name=name,
        description=description,
        function=func,
        schema=schema,
    ))

    # ----------------------------------------------------------
    # Paso 4: Devolver la función original sin modificar.
    # ----------------------------------------------------------
    # El decorador NO envuelve ni transforma la función: solo la registra
    # como efecto secundario. Esto tiene dos ventajas:
    #   1. La función puede seguir usándose como una función Python normal
    #      en tests unitarios u otros contextos sin pasar por el registro.
    #   2. No hay sobrecarga de rendimiento en llamadas directas.
    return func
