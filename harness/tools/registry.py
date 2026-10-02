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

import inspect  # Introspección de funciones (docstrings, firmas).
from collections.abc import Callable  # Tipo genérico para funciones invocables.
from dataclasses import (
    dataclass,  # Generación de clases de datos con boilerplate mínimo.
)
from typing import Any  # Tipo comodín para argumentos de despacho dinámico.

from pydantic import (
    TypeAdapter,  # Generación de esquemas JSON a partir de tipos Python.
)


@dataclass
class Tool:
    """
    Representa una herramienta registrada.

    Agrupa toda la información necesaria para exponer una función Python
    como herramienta del agente: su identidad (nombre), su documentación
    (descripción), la referencia a la función ejecutable y el esquema JSON
    de sus parámetros.

    Attributes:
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
    name: str
    description: str
    function: Callable
    schema: dict


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
        durante el despacho sea O(1) en lugar de O(n).
        """
        # Diccionario interno que mapea nombre de herramienta → objeto Tool.
        # El guion bajo indica que es un atributo privado: el acceso externo
        # debe hacerse a través de los métodos register(), get_schemas()
        # y dispatch().
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
        # Última escritura gana: si el mismo nombre se registra dos veces,
        # la segunda definición reemplaza silenciosamente a la primera.
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
                    "parameters": { ... }  # esquema JSON
                }
            }

        Este método construye esa estructura para cada herramienta
        registrada y la devuelve como lista. Se llama en cada petición
        al modelo para que conozca las herramientas disponibles.

        Returns:
            Lista de diccionarios, uno por herramienta registrada, en el
            formato que espera ``client.chat.completions.create(tools=...)``.
        """
        # Envolvemos el esquema de cada herramienta en la envoltura
        # {"type": "function", "function": {...}} que exige la API de OpenAI.
        # El campo "parameters" contiene el esquema JSON generado por
        # Pydantic, que describe los tipos, nombres y obligatoriedad de
        # cada parámetro de la función.
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                },
            }
            for t in self._tools.values()
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
        """
        # Rechazar nombres de herramienta desconocidos. Devolvemos un string
        # de error (en lugar de lanzar una excepción) para que el modelo
        # pueda leerlo y recuperarse: por ejemplo, corrigiendo el nombre o
        # informando al usuario de que la herramienta no existe.
        if name not in self._tools:
            return f"error: herramienta desconocida '{name}'"

        # Ejecutar la herramienta dentro de un bloque try/except para que
        # cualquier excepción (FileNotFoundError, PermissionError, ValueError,
        # etc.) no propague hacia arriba y rompa el bucle del agente.
        # El modelo recibe el string de error y decide el siguiente paso.
        try:
            # Desempaquetamos el diccionario de argumentos como keyword
            # arguments de la función. Por ejemplo, si arguments es
            # {"path": "src/main.py"}, la llamada equivale a
            # function(path="src/main.py").
            result = self._tools[name].function(**arguments)

            # Convertimos el resultado a string para que pueda insertarse
            # en el mensaje de rol "tool" del historial. Las funciones de
            # herramientas ya devuelven strings, pero esta conversión es una
            # red de seguridad por si alguna devuelve otro tipo.
            return str(result)

        except Exception as e:
            # Capturamos cualquier excepción y la formateamos como un string
            # legible que incluye el tipo de excepción y su mensaje.
            # Ejemplo: "error: FileNotFoundError: [Errno 2] No such file..."
            return f"error: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Instancia global del registro
# ---------------------------------------------------------------------------
# Este es el único registro que usa todo el harness. Los módulos de
# herramientas importan `tool` (el decorador) para registrarse, y el
# agente importa `registry` para obtener esquemas y despachar llamadas.
# Se crea en el momento del import de este módulo.
registry = ToolRegistry()


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

    Example::

        @tool
        def read(path: str) -> str:
            '''Lee un fichero del workspace.'''
            return Path(path).read_text()

        # Tras el import, registry contiene una herramienta llamada "read"
        # con el docstring como descripción y {"path": {"type": "string"}}
        # como esquema de parámetros.
    """
    # Paso 1: Extraer metadatos de la función.
    # El nombre se usa como identificador único en el registro y como el
    # nombre que el modelo verá en la lista de herramientas disponibles.
    name = func.__name__

    # El docstring se usa como descripción para el modelo. `inspect.getdoc()`
    # es preferible a `func.__doc__` porque limpia la indentación heredada
    # del código fuente. Si la función no tiene docstring, usamos una
    # cadena vacía para evitar None.
    description = inspect.getdoc(func) or ""

    # Paso 2: Generar el esquema JSON de los parámetros de la función.
    # TypeAdapter de Pydantic introspecciona la firma de la función (nombres
    # de parámetros, anotaciones de tipo, valores por defecto) y produce un
    # esquema JSON compatible con el formato "parameters" de OpenAI.
    # Por ejemplo, para `def read(path: str) -> str:` genera algo como:
    #   {
    #     "type": "object",
    #     "properties": {"path": {"type": "string"}},
    #     "required": ["path"]
    #   }
    schema = TypeAdapter(func).json_schema()

    # Paso 3: Crear el objeto Tool con todos los metadatos y registrarlo
    # en la instancia global del registro. A partir de este momento, la
    # herramienta aparece en get_schemas() y puede ser despachada por nombre.
    registry.register(Tool(
        name=name,
        description=description,
        function=func,
        schema=schema,
    ))

    # Paso 4: Devolver la función original sin modificar. El decorador no
    # envuelve ni transforma la función: solo la registra como efecto
    # secundario. Esto permite que la función pueda seguir usándose como
    # una función Python normal en tests u otros contextos.
    return func
