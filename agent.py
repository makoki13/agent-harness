# ============================================================
# IMPORTACIÓN DE MÓDULOS
# ============================================================

# Módulo estándar para interactuar con el sistema operativo
# (leer variables de entorno, manejar rutas, etc.)
import os

# load_dotenv: carga variables de entorno desde un archivo .env
# situado en el directorio del proyecto, facilitando la gestión
# de claves API sin exponerlas en el código fuente.
from dotenv import load_dotenv

# Cliente de la API de OpenAI (compatible también con proveedores
# que implementen la misma interfaz, como Groq).
from openai import OpenAI

# Tipo para el tipado estático de los mensajes de chat.
from openai.types.chat import ChatCompletionMessageParam

# ============================================================
# CARGA DE VARIABLES DE ENTORNO
# ============================================================
# Busca un archivo .env en el directorio actual y carga las
# variables definidas en él (por ejemplo, GROQ_API_KEY) como
# variables de entorno del proceso.
load_dotenv()

# ============================================================
# CONFIGURACIÓN DEL CLIENTE Y DEL MODELO
# ============================================================
# Si existe la variable de entorno GROQ_API_KEY, usamos Groq
# como proveedor de inferencia (compatible con la API de OpenAI).
# En caso contrario, se usa la API oficial de OpenAI.
if os.getenv("GROQ_API_KEY"):
    # Modelo a utilizar en Groq (formato: proveedor/nombre-modelo)
    MODEL = "qwen/qwen3.8-27b"
    # Se instancia el cliente OpenAI apuntando al endpoint de Groq.
    # Groq expone una API compatible con el formato de OpenAI,
    # por lo que podemos reutilizar el mismo SDK cambiando la URL base.
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    # Diccionario para parámetros adicionales del cuerpo de la petición.
    # Se deja vacío porque Groq no requiere parámetros extra aquí.
    EXTRA_BODY = {}
else:
    # Modelo por defecto si usamos la API de OpenAI directamente.
    MODEL = "gpt-4o-mini"
    # Se instancia el cliente OpenAI; la clave API se toma
    # automáticamente de la variable de entorno OPENAI_API_KEY.
    client = OpenAI()
    EXTRA_BODY = {}


# ============================================================
# FUNCIÓN PRINCIPAL: BUCLE DE CONVERSACIÓN DEL AGENTE
# ============================================================
def run():
    '''Ejecuta el bucle de conversación del agente hasta que el usuario decida salir.'''

    # Prompt de sistema: define el comportamiento, las capacidades
    # y las reglas que el modelo debe seguir durante toda la conversación.
    SYSTEM_PROMPT = '''
Eres un asistente de programación que se ejecuta en una terminal, ayudando a un desarrollador con tareas de ingeniería de software.

Sé conciso. Prefiere respuestas cortas y directas frente a respuestas largas. Cuando el usuario pida código, devuelve el código con una explicación mínima, a menos que pidan más detalles.

Cuando devuelvas código, usa bloques de código delimitados (fenced code blocks) y especifica el lenguaje.

Tienes acceso a cinco herramientas de sistema de archivos — leer, escribir, listar, crear directorio, eliminar — que operan sobre un directorio de trabajo. Úsalas siempre que una tarea implique leer, modificar u organizar archivos. Las rutas son relativas a la raíz del directorio de trabajo. Prefiere leer y escribir archivos reales en lugar de describirlos en la conversación.

También dispones de seis herramientas de git — git_status, git_diff, git_log, git_commit, git_checkout, git_branch — para versionar tu trabajo. El directorio de trabajo ya está inicializado como un repositorio git. Usa git para:
- Hacer commits con frecuencia. Los commits pequeños y enfocados son más fáciles de revertir.
- Hacer commit antes de realizar cualquier acción arriesgada (reescrituras grandes, eliminación de archivos, reestructuraciones). Un commit antes del paso arriesgado te proporciona un punto de recuperación.
- Escribir mensajes de commit significativos: describe qué cambió y por qué, en tiempo presente (por ejemplo, "añadir módulo de autenticación de usuario").
- Ramificar experimentos. Cuando pruebes un enfoque alternativo, crea primero una rama para que la línea principal de trabajo permanezca intacta.

Tienes una herramienta más: `bash`. Ejecuta comandos de shell en el directorio de trabajo con interpretación completa del shell — tuberías (pipes), redirecciones y encadenamiento de comandos funcionan correctamente. Usa bash para cualquier cosa que las herramientas específicas anteriores no cubran: ejecutar scripts (python foo.py, node foo.js), invocar utilidades del sistema (grep, find, curl, wc, sort, awk), instalar paquetes (pip install ...), o explorar el entorno (ls, pwd, which python).

Prefiere las herramientas específicas cuando sean aplicables. Si la tarea es leer un archivo, usa `read`, no `bash("cat archivo.md")`. Si la tarea es hacer un commit, usa `git_commit`, no `bash("git commit ...")`. Las herramientas específicas son más seguras, más rápidas y más fáciles de trazar. Recurre a `bash` cuando las herramientas específicas no cubran lo que necesitas — lo cual sucede a menudo, porque el trabajo de software es variado.

Los comandos de bash ven el directorio de trabajo como su directorio actual. Un `cd` dentro de un comando bash no persiste hasta la siguiente llamada a herramienta; cada invocación de bash comienza de nuevo desde la raíz del directorio de trabajo.

El directorio de trabajo contiene un archivo `AGENTS.md` — tu memoria persistente entre sesiones. Se carga automáticamente en tu contexto al inicio de cada sesión. Actualízalo (usando la herramienta `write`) cuando aprendas algo que merezca ser recordado para futuras sesiones. Cosas útiles que escribir:
- Contexto del proyecto: qué es este código, qué hace, quién lo usa
- Convenciones que has observado: estilo de código, bibliotecas, patrones de nomenclatura
- Decisiones que se han tomado y el razonamiento detrás de ellas
- Trampas: peculiaridades, dependencias no obvias, cosas que han causado problemas en sesiones anteriores
- Tareas activas: en qué se está trabajando actualmente (elimínalas cuando se completen)

Cuando actualices AGENTS.md, preserva la estructura existente (los encabezados de sección). Añade contenido a la sección relevante en lugar de reemplazar todo el archivo. Si la sección comienza con una pista entre paréntesis como "(¿Qué es este proyecto?)", reemplaza la pista con contenido real a medida que lo vayas completando.
    '''  # noqa: N806

    # Historial de la conversación. Esta es la memoria completa del agente.
    # En cada turno, añadimos mensajes nuevos y enviamos el historial
    # completo al modelo para que mantenga el contexto de la conversación.
    # Se inicializa con el mensaje de sistema que define el comportamiento.
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    # Mensaje de bienvenida que indica al usuario que el agente está listo.
    print("Agente listo. Escribe 'quit' o 'exit' para salir.\n")

    # Bucle principal: se repite indefinidamente hasta que el usuario salga.
    while True:
        # 1. Obtener la entrada del usuario desde la terminal.
        #    Se usa .strip() para eliminar espacios en blanco al inicio y al final.
        user_input = input("tú > ").strip()

        # 2. Permitir al usuario salir limpiamente del bucle.
        #    Si escribe "quit" o "exit", se rompe el bucle y termina el programa.
        if user_input in {"quit", "exit"}:
            print("¡Hasta luego!")
            break

        # Si el usuario pulsa Enter sin escribir nada, saltamos esta iteración
        # sin realizar ninguna llamada al modelo (evita peticiones innecesarias).
        if not user_input:
            continue

        # 3. Añadir el mensaje del usuario al historial de conversación.
        #    El rol "user" indica que este mensaje proviene del usuario.
        messages.append({
            "role": "user",
            "content": user_input
        })

        # 4. Llamar al modelo con la conversación completa hasta el momento.
        #    Se envía todo el historial (system + user + assistant) para que
        #    el modelo tenga contexto completo y pueda responder coherentemente.
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
        )

        # 5. Extraer la respuesta del asistente del objeto de respuesta.
        #    response.choices[0] es la primera (y única) respuesta generada.
        #    .message.content contiene el texto generado por el modelo.
        assistant_message = response.choices[0].message.content

        # 6. Añadir la respuesta del asistente al historial de conversación.
        #    Esto permite que en turnos futuros el modelo "recuerde"
        #    lo que respondió anteriormente.
        messages.append({"role": "assistant", "content": assistant_message})

        # 7. Mostrar la respuesta del agente al usuario en la terminal.
        print(f"\nagente > {assistant_message}")


# ============================================================
# PUNTO DE ENTRADA DEL PROGRAMA
# ============================================================
# Este bloque garantiza que la función run() solo se ejecute cuando
# el script se ejecuta directamente (python script.py), y no cuando
# se importa como módulo desde otro archivo.
if __name__ == "__main__":
    run()
