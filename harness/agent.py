"""
agent.py — Punto de entrada del agente conversacional con herramientas.

Este módulo implementa el bucle principal de interacción entre el usuario
y un modelo de lenguaje. El agente:

  1. Recibe instrucciones del usuario desde la terminal.
  2. Envía la conversación completa (historial + mensaje nuevo) al modelo,
     junto con los esquemas de las herramientas disponibles.
  3. Si el modelo solicita ejecutar una o varias herramientas, se despachan,
     se añaden los resultados al historial y se vuelve a llamar al modelo
     para obtener la respuesta final en texto.
  4. Muestra la respuesta al usuario y repite.

El historial de mensajes (`messages`) actúa como la única memoria del agente:
se envía completo en cada petición al modelo. No hay memoria externa ni
persistencia entre sesiones (salvo lo que el modelo escriba en AGENTS.md
mediante las herramientas de filesystem).

Uso:
    python agent.py

Dependencias:
    - openai        → SDK para hablar con APIs compatibles con OpenAI.
    - python-dotenv → Carga variables de entorno desde un fichero .env.
    - harness.tools → Paquete interno que contiene el registro de herramientas
                      y las implementaciones concretas (filesystem, git, bash…).
"""

import json  # noqa: I001  # Necesario para deserializar los argumentos de las llamadas a herramientas.
import os                   # Lectura de variables de entorno para la configuración del backend.

from dotenv import load_dotenv  # Carga el fichero .env en las variables de entorno del proceso.
from openai import OpenAI       # Cliente HTTP para la API de chat completions.

# Importamos la instancia global del registro de herramientas directamente
# desde su módulo para evitar la ambigüedad de nombres entre el módulo
# `harness.tools.registry` y la variable `registry` que contiene.
from harness.tools.registry import registry

# Importamos el módulo de herramientas de filesystem por sus efectos secundarios:
# cada función decorada con @tool se registra automáticamente en `registry`
# al ejecutarse el import. No usamos ningún nombre de este módulo directamente,
# por eso el noqa.
from harness.tools import filesystem  # noqa: F401
from harness.tools import git         # noqa: F401

# Carga las variables definidas en el fichero .env (si existe) en os.environ.
# Esto permite configurar claves API y URLs sin modificar el código fuente.
load_dotenv()

# ---------------------------------------------------------------------------
# Selección del backend de inferencia
# ---------------------------------------------------------------------------
# La elección se hace en tiempo de configuración: si existe la variable de
# entorno GROQ_API_KEY, se usa Groq; en caso contrario, se asume OpenAI.
# Para cambiar de proveedor basta con editar el fichero .env, sin tocar código.

if os.getenv("GROQ_API_KEY"):
    # Modelo a utilizar en Groq.
    # Formato: "proveedor/nombre-modelo", tal como lo espera la API de Groq.
    MODEL = "qwen/qwen3.8-27b"

    # Se instancia el cliente OpenAI apuntando al endpoint de Groq.
    # Groq expone una API compatible con el formato de OpenAI,
    # por lo que podemos reutilizar el mismo SDK cambiando la URL base.
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )

    # Diccionario para parámetros adicionales que se inyectan en el cuerpo
    # de cada petición HTTP. Se deja vacío porque Groq no requiere
    # parámetros extra en este caso. Otros proveedores (p. ej. Kimi) podrían
    # necesitar opciones como {"thinking": {"type": "disabled"}}.
    EXTRA_BODY = {}

else:
    # Modelo por defecto si usamos la API de OpenAI directamente.
    MODEL = "gpt-4o-mini"

    # Se instancia el cliente OpenAI sin argumentos: la clave API se toma
    # automáticamente de la variable de entorno OPENAI_API_KEY y la URL base
    # apunta a https://api.openai.com/v1 por defecto.
    client = OpenAI()

    # Sin parámetros extra para OpenAI.
    EXTRA_BODY = {}

# ---------------------------------------------------------------------------
# Prompt de sistema
# ---------------------------------------------------------------------------
# Define la personalidad, el tono y las capacidades del agente. Se envía
# como primer mensaje del historial en cada petición para que el modelo
# mantenga el contexto de su rol durante toda la conversación.

SYSTEM_PROMPT = """
Eres un asistente de programación que se ejecuta en una terminal, ayudando a un desarrollador con tareas de ingeniería de software.

Sé conciso. Prefiere respuestas cortas y directas frente a respuestas largas. Cuando el usuario pida código, devuelve el código con una explicación mínima salvo que pidan más detalle.

Cuando devuelvas código, usa bloques de código delimitados con triple backtick e indica el lenguaje.

Tienes acceso a cinco herramientas de sistema de ficheros — read, write, list, mkdir, delete — que operan sobre un directorio de trabajo. Úsalas siempre que una tarea implique leer, modificar u organizar ficheros. Las rutas son relativas a la raíz del espacio de trabajo. Prefiere leer y escribir ficheros reales antes que describirlos en la conversación.

You also have six git tools — git_status, git_diff, git_log, git_commit, git_checkout, git_branch — for versioning your work. The workspace is
already initialized as a git repo. Use git to:
- Commit frequently. Small, focused commits are easier to roll back.
- Commit before doing anything risky (large rewrites, deleting files, restructuring). A commit before the risky step gives you a recovery point.
- Write meaningful commit messages — describe what changed and why, in the present tense (e.g., "add user authentication module").
- Branch experiments. When trying an alternative approach, create a branch first so the main line of work stays intact.
"""


def run() -> None:
    """
    Bucle principal del agente.

    Mantiene una conversación interactiva con el usuario hasta que este
    escribe 'quit' o 'exit'. En cada turno:

      1. Lee la entrada del usuario.
      2. La añade al historial de mensajes.
      3. Envía el historial completo + herramientas al modelo.
      4. Si el modelo responde con llamadas a herramientas, las ejecuta,
         añade los resultados al historial y vuelve a consultar al modelo.
      5. Muestra la respuesta textual final al usuario.

    El historial (`messages`) crece de forma ilimitada durante la sesión.
    Es la única memoria del agente: no hay resumen ni ventana deslizante.
    """

    # Historial de la conversación. Es la memoria completa del agente.
    # Cada turno añadimos mensajes y enviamos la lista entera al modelo.
    # El primer mensaje es siempre el prompt de sistema, que establece
    # el rol y las instrucciones permanentes del asistente.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("Agente listo. Escribe 'quit' o 'exit' para salir.\n")

    while True:

        # ------------------------------------------------------------------
        # Paso 1: Leer la entrada del usuario desde la terminal.
        # ------------------------------------------------------------------
        # Se usa .strip() para eliminar espacios en blanco y saltos de línea
        # sobrantes que el usuario pudiera introducir accidentalmente.
        user_input = input("tú > ").strip()

        # ------------------------------------------------------------------
        # Paso 2: Comprobar si el usuario quiere salir del bucle.
        # ------------------------------------------------------------------
        # Se aceptan ambas palabras clave para mayor comodidad.
        if user_input in {"quit", "exit"}:
            print("Hasta luego.")
            break

        # Si el usuario pulsa Enter sin escribir nada, saltamos la iteración
        # sin hacer una llamada al modelo (evita peticiones vacías innecesarias).
        if not user_input:
            continue

        # ------------------------------------------------------------------
        # Paso 3: Añadir el mensaje del usuario al historial.
        # ------------------------------------------------------------------
        # El rol "user" indica al modelo que este mensaje proviene del
        # usuario humano (a diferencia de "system" o "assistant").
        messages.append({
            "role": "user",
            "content": user_input,
        })

        # ------------------------------------------------------------------
        # Paso 4: Primera llamada al modelo.
        # ------------------------------------------------------------------
        # Enviamos el historial completo de la conversación junto con los
        # esquemas JSON de todas las herramientas registradas. El modelo
        # puede responder de dos formas:
        #   a) Texto directo (message.tool_calls es None) → vamos al paso 7.
        #   b) Una o más llamadas a herramientas → entramos en el bloque
        #      de despacho de herramientas (pasos 5-6).
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,  # pyright: ignore[reportArgumentType]
            extra_body=EXTRA_BODY,
            tools=registry.get_schemas(),  # pyright: ignore[reportArgumentType]
        )

        # Extraemos el mensaje del modelo de la respuesta.
        # `message` puede contener texto (message.content), llamadas a
        # herramientas (message.tool_calls), o ambos.
        message = response.choices[0].message

        # ------------------------------------------------------------------
        # Paso 5: Despacho de herramientas (si el modelo las solicitó).
        # ------------------------------------------------------------------
        # Si el modelo decidió que necesita ejecutar herramientas para
        # responder, `message.tool_calls` contiene una lista de objetos
        # con el nombre de la función y sus argumentos serializados en JSON.
        if message.tool_calls:

            # Paso 5a: Registrar el mensaje del modelo (que contiene las
            # llamadas a herramientas) en el historial. Esto es necesario
            # porque los mensajes de resultado de herramienta (role="tool")
            # deben referenciar un tool_call_id que exista en el historial.
            messages.append(message)  # pyright: ignore[reportArgumentType]

            # Paso 5b: Ejecutar cada herramienta solicitada y añadir su
            # resultado al historial. El campo tool_call_id vincula el
            # resultado con la llamada original para que el modelo pueda
            # emparejarlos correctamente.
            for call in message.tool_calls:
                # Los argumentos vienen como un string JSON (p. ej. '{"path": "src/main.py"}').
                # Los deserializamos a un dict para pasarlos como kwargs a la función.
                arguments = json.loads(call.function.arguments)  # pyright: ignore[reportAttributeAccessIssue]

                # Despachamos la llamada al registro de herramientas.
                # `registry.dispatch` busca la función por nombre, la ejecuta
                # con los argumentos dados y devuelve el resultado como string.
                # Si la herramienta lanza una excepción, dispatch la captura
                # y devuelve un mensaje de error legible por el modelo.
                result = registry.dispatch(call.function.name, arguments)  # pyright: ignore[reportAttributeAccessIssue]

                # Añadimos el resultado como un mensaje de rol "tool".
                # El modelo usa tool_call_id para saber a qué llamada
                # corresponde este resultado.
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

            # Paso 6: Segunda llamada al modelo con los resultados.
            # ------------------------------------------------------------------
            # Ahora que el historial contiene las respuestas de las
            # herramientas, volvemos a consultar al modelo para que genere
            # la respuesta textual final dirigida al usuario. El modelo
            # puede decidir que necesita más herramientas (en cuyo caso
            # habría que repetir el ciclo), pero en esta implementación
            # mínima solo se hace una ronda de despacho por turno.
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,  # pyright: ignore[reportArgumentType]
                tools=registry.get_schemas(),  # pyright: ignore[reportArgumentType]
            )
            message = response.choices[0].message

        # ------------------------------------------------------------------
        # Paso 7: Extraer y mostrar la respuesta final al usuario.
        # ------------------------------------------------------------------
        # Llegados aquí, `message` contiene la respuesta textual definitiva
        # del modelo para este turno, ya sea:
        #   - Directa (el modelo no necesitó herramientas), o
        #   - Tras el despacho de herramientas (segunda llamada).
        assistant_text = message.content

        # Registramos la respuesta del asistente en el historial para que
        # esté disponible como contexto en turnos futuros.
        messages.append({"role": "assistant", "content": assistant_text})  # pyright: ignore[reportArgumentType]

        # Mostramos la respuesta en la terminal con un prefijo identificativo.
        print(f"\nagente > {assistant_text}\n")


# ---------------------------------------------------------------------------
# Punto de entrada del script.
# ---------------------------------------------------------------------------
# Permite ejecutar el agente directamente con `python agent.py` sin que
# el bucle se inicie si el módulo se importa desde otro lugar.
if __name__ == "__main__":
    run()
