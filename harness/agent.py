# ============================================================
# IMPORTACIÓN DE MÓDULOS
# ============================================================

# json: se usa para deserializar los argumentos que el modelo envía
# en las llamadas a herramientas (vienen como cadena JSON).
import json

# os: permite leer variables de entorno del sistema operativo.
import os

# load_dotenv: carga variables de entorno desde un archivo .env
# situado en el directorio del proyecto. Esto evita tener que exportar
# las claves API manualmente o escribirlas en el código fuente.
from dotenv import load_dotenv

# Cliente de la API de OpenAI (compatible también con proveedores
# que implementen la misma interfaz, como Groq).
from openai import OpenAI

# agents_md: módulo que gestiona el archivo AGENTS.md, la memoria
# persistente del agente entre sesiones. Se carga al inicio de cada
# conversación y se actualiza cuando el agente aprende algo relevante.
from harness.memory import agents_md

# registry: registro central de herramientas. Las funciones decoradas
# con @tool en otros módulos se registran aquí automáticamente.
# Proporciona métodos para obtener los esquemas JSON (para enviarlos
# al modelo) y para despachar (ejecutar) una herramienta por nombre.
from harness.tools.registry import registry

# ============================================================
# CARGA DE VARIABLES DE ENTORNO
# ============================================================
# Busca un archivo .env en el directorio actual y carga las variables
# definidas en él (por ejemplo, GROQ_API_KEY u OPENAI_API_KEY) como
# variables de entorno del proceso actual.
load_dotenv()

# ============================================================
# CONFIGURACIÓN DEL BACKEND (PROVEEDOR DE INFERENCIA)
# ============================================================
# Se decide qué backend utilizar en función de qué clave API está
# definida en el archivo .env. Es una decisión en tiempo de configuración:
# para cambiar de proveedor, se modifica el .env, no el código.

if os.getenv("GROQ_API_KEY"):
    # --- PROVEEDOR: GROQ ---
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
    # --- PROVEEDOR: OPENAI (por defecto) ---
    # Modelo por defecto si usamos la API de OpenAI directamente.
    MODEL = "gpt-4o-mini"

    # Se instancia el cliente OpenAI sin argumentos: la clave API se toma
    # automáticamente de la variable de entorno OPENAI_API_KEY y la URL base
    # apunta a https://api.openai.com/v1 por defecto.
    client = OpenAI()

    # Sin parámetros extra para OpenAI.
    EXTRA_BODY = {}


# ============================================================
# PROMPT DE SISTEMA
# ============================================================
# Define el comportamiento, las capacidades y las reglas que el modelo
# debe seguir durante toda la conversación. Se envía como primer mensaje
# en cada petición al modelo para que siempre tenga presente su rol.

SYSTEM_PROMPT = """
Eres un asistente de programación que se ejecuta en una terminal, ayudando a un desarrollador con tareas de ingeniería de software.

Sé conciso. Prefiere respuestas cortas y directas frente a respuestas largas. Cuando el usuario pida código, devuelve el código con una explicación mínima, a menos que pidan más detalles.

Cuando devuelvas código, usa bloques de código delimitados (fenced code blocks) y especifica el lenguaje.

Tienes acceso a cinco herramientas de sistema de archivos — read, write, list, mkdir, delete — que operan sobre un directorio de trabajo. Úsalas siempre que una tarea implique leer, modificar u organizar archivos. Las rutas son relativas a la raíz del directorio de trabajo. Prefiere leer y escribir archivos reales en lugar de describirlos en la conversación.

También dispones de seis herramientas de git — git_status, git_diff, git_log, git_commit, git_checkout, git_branch — para versionar tu trabajo. El directorio de trabajo ya está inicializado como un repositorio git. Usa git para:
- Hacer commits con frecuencia. Los commits pequeños y enfocados son más fáciles de revertir.
- Hacer commit antes de realizar cualquier acción arriesgada (reescrituras grandes, eliminación de archivos, reestructuraciones). Un commit antes del paso arriesgado te proporciona un punto de recuperación.
- Escribir mensajes de commit significativos: describe qué cambió y por qué, en tiempo presente (por ejemplo, "añadir módulo de autenticación de usuario").
- Ramificar experimentos. Cuando pruebes un enfoque alternativo, crea primero una rama para que la línea principal de trabajo permanezca intacta.

El directorio de trabajo contiene un archivo `AGENTS.md` — tu memoria persistente entre sesiones. Se carga automáticamente en tu contexto al inicio de cada sesión. Actualízalo (usando la herramienta `write`) cuando aprendas algo que merezca ser recordado para futuras sesiones. Cosas útiles que escribir:
- Contexto del proyecto: qué es este código, qué hace, quién lo usa
- Convenciones que has observado: estilo de código, bibliotecas, patrones de nomenclatura
- Decisiones que se han tomado y el razonamiento detrás de ellas
- Trampas: peculiaridades, dependencias no obvias, cosas que han causado problemas en sesiones anteriores
- Tareas activas: en qué se está trabajando actualmente (elimínalas cuando se completen)

Cuando actualices AGENTS.md, preserva la estructura existente (los encabezados de sección). Añade contenido a la sección relevante en lugar de reemplazar todo el archivo. Si la sección comienza con una pista entre paréntesis como "(¿Qué es este proyecto?)", reemplaza la pista con contenido real a medida que lo vayas completando.
"""


# ============================================================
# FUNCIÓN PRINCIPAL: BUCLE DE CONVERSACIÓN DEL AGENTE
# ============================================================

def run() -> None:
    """Ejecuta el bucle de conversación del agente hasta que el usuario decida salir."""

    # ----------------------------------------------------------
    # CARGA DE LA MEMORIA PERSISTENTE Y CONSTRUCCIÓN DEL HISTORIAL
    # ----------------------------------------------------------
    # Se carga el archivo AGENTS.md (memoria acumulada del proyecto)
    # y se ensambla la lista inicial de mensajes.
    # El primer mensaje de sistema es el prompt del harness (comportamiento);
    # el segundo es la memoria acumulada del proyecto (contexto persistente).

    # Historial de la conversación. Esta es la memoria completa del agente.
    # En cada turno, añadimos mensajes nuevos y enviamos el historial
    # completo al modelo para que mantenga el contexto de la conversación.
    messages = [
        # Primer mensaje de sistema: define el rol y las reglas del agente.
        {"role": "system", "content": SYSTEM_PROMPT},
        # Segundo mensaje de sistema: inyecta la memoria persistente del proyecto
        # (contenido de AGENTS.md) para que el agente tenga contexto acumulado.
        {"role": "system", "content": agents_md.load_agents_md()},
    ]

    # Mensaje de bienvenida que indica al usuario que el agente está operativo.
    print("Agente listo. Escribe 'quit' o 'exit' para salir.\n")

    # ----------------------------------------------------------
    # BUCLE PRINCIPAL DE CONVERSACIÓN
    # ----------------------------------------------------------
    # Se repite indefinidamente hasta que el usuario escriba "quit" o "exit".
    while True:

        # 1. Obtener la entrada del usuario desde la terminal.
        #    .strip() elimina espacios en blanco al inicio y al final.
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
            "content": user_input,
        })

        # ----------------------------------------------------------
        # 4. PRIMERA LLAMADA AL MODELO
        # ----------------------------------------------------------
        # Se envía al modelo:
        #   - El historial completo de la conversación (messages)
        #   - Parámetros extra del proveedor (extra_body)
        #   - Los esquemas JSON de todas las herramientas disponibles
        #     (tools), para que el modelo sepa qué puede invocar.
        #
        # El modelo puede responder de dos formas:
        #   a) Con texto directo (sin tool_calls) → respuesta final.
        #   b) Con una o más llamadas a herramientas (tool_calls) →
        #      el agente debe ejecutarlas y volver a llamar al modelo.
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,           # pyright: ignore[reportArgumentType]
            extra_body=EXTRA_BODY,
            tools=registry.get_schemas() # pyright: ignore[reportArgumentType]
        )

        # Extraer el mensaje del asistente de la respuesta.
        message = response.choices[0].message

        # ----------------------------------------------------------
        # 5. GESTIÓN DE LLAMADAS A HERRAMIENTAS (TOOL CALLING)
        # ----------------------------------------------------------
        # Si el modelo solicitó ejecutar una o más herramientas,
        # se procesan antes de generar la respuesta final al usuario.
        # Implementación mínima viable: una sola ronda de herramientas
        # (el modelo no puede encadenar múltiples rondas en un mismo turno).
        if message.tool_calls:

            # Paso 1: Registrar el mensaje del modelo con las llamadas a
            # herramientas en el historial. Esto es necesario para que los
            # mensajes de resultado de herramientas que añadiremos a
            # continuación tengan un mensaje al que hacer referencia
            # (mediante tool_call_id).
            messages.append(message)  # pyright: ignore[reportArgumentType]

            # Paso 2: Ejecutar cada herramienta solicitada y añadir su
            # resultado al historial. Se usa el tool_call_id correspondiente
            # para que el modelo pueda emparejar cada resultado con su solicitud.
            for call in message.tool_calls:
                # Deserializar los argumentos JSON que el modelo envió.
                # call.function.arguments es una cadena JSON, por ejemplo:
                # '{"path": "src/main.py", "content": "print(42)"}'
                arguments = json.loads(call.function.arguments)  # pyright: ignore[reportAttributeAccessIssue]

                # Despachar (ejecutar) la herramienta por nombre, pasándole
                # los argumentos deserializados. registry.dispatch busca la
                # función registrada con ese nombre y la invoca.
                result = registry.dispatch(call.function.name, arguments)  # pyright: ignore[reportAttributeAccessIssue]

                # Añadir el resultado de la herramienta al historial.
                # El rol "tool" indica que este mensaje es el resultado
                # de una ejecución de herramienta. El tool_call_id vincula
                # este resultado con la solicitud original del modelo.
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": result,
                })

            # Paso 3: Volver a llamar al modelo, ahora con los resultados
            # de las herramientas en el contexto. Esta segunda llamada
            # produce la respuesta textual final del modelo para este turno,
            # basada en la información obtenida de las herramientas.
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,           # pyright: ignore[reportArgumentType]
                tools=registry.get_schemas() # pyright: ignore[reportArgumentType]
            )
            # Extraer el mensaje final del asistente de la segunda respuesta.
            message = response.choices[0].message

        # ----------------------------------------------------------
        # 6. EXTRAER Y REGISTRAR LA RESPUESTA FINAL
        # ----------------------------------------------------------
        # En este punto, `message` es la respuesta textual final del modelo
        # para este turno. Puede provenir de:
        #   - La primera llamada (si no se necesitaron herramientas), o
        #   - La segunda llamada (tras ejecutar las herramientas).
        #
        # Si el modelo usó herramientas pero no generó texto adicional,
        # se usa un mensaje de marcador para informar al usuario.
        assistant_text = message.content or "(sin respuesta textual: solo se usaron herramientas)"

        # Añadir la respuesta del asistente al historial para que esté
        # disponible como contexto en futuros turnos de la conversación.
        messages.append({"role": "assistant", "content": assistant_text})

        # 7. Mostrar la respuesta del agente al usuario en la terminal.
        print(f"\nagente > {assistant_text}\n")


# ============================================================
# PUNTO DE ENTRADA DEL PROGRAMA
# ============================================================
# Este bloque garantiza que la función run() solo se ejecute cuando
# el script se ejecuta directamente (python script.py), y no cuando
# se importa como módulo desde otro archivo.
if __name__ == "__main__":
    run()
