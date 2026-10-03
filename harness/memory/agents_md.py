"""
Memoria entre sesiones: el patrón AGENTS.md.

El harness (entorno de ejecución del agente) lee el archivo AGENTS.md
del directorio de trabajo al inicio de cada sesión e inyecta su contenido
como un segundo mensaje de sistema en la conversación. El agente utiliza
su herramienta `write` ya existente para actualizar el archivo a medida
que aprende cosas nuevas.

Este módulo proporciona:
  - La ruta donde se almacena el archivo de memoria.
  - Una plantilla inicial con secciones guiadas.
  - Una función para cargar (o crear) el archivo y devolver su contenido.
"""

# ============================================================
# IMPORTACIONES
# ============================================================

# WORKSPACE: constante (objeto pathlib.Path) que apunta al directorio
# de trabajo del agente. Todos los archivos del proyecto viven dentro
# de este directorio. Se importa desde el módulo de herramientas de
# sistema de archivos para garantizar que la memoria se almacena en
# el mismo espacio que el código del proyecto.
from harness.tools.filesystem import WORKSPACE

# ============================================================
# RUTA DEL ARCHIVO DE MEMORIA
# ============================================================

# Ruta única y fija del archivo de memoria del workspace.
# Se construye concatenando el directorio de trabajo con el nombre
# del archivo. Ejemplo: /home/user/proyecto/AGENTS.md
AGENTS_MD_PATH = WORKSPACE / "AGENTS.md"


# ============================================================
# PLANTILLA INICIAL DEL ARCHIVO DE MEMORIA
# ============================================================
# Esta plantilla se escribe en el archivo AGENTS.md cuando este aún
# no existe (es decir, la primera vez que el agente se ejecuta en un
# workspace nuevo).
#
# Los encabezados de sección y las pistas entre paréntesis cumplen una
# doble función:
#   - Al LEER: el modelo sabe para qué sirve cada sección y puede
#     localizar información relevante rápidamente.
#   - Al ESCRIBIR: el modelo sabe qué tipo de contenido pertenece a
#     cada sección y dónde debe añadirlo.
#
# La barra invertida "\" después de las comillas triples ("""\) evita
# que se incluya una línea en blanco al inicio del string.

AGENTS_MD_TEMPLATE = """\
# Memoria del proyecto

> Este archivo es la memoria duradera del agente entre sesiones. El harness
> lo carga al inicio de cada sesión. Actualízalo cada vez que aprendas algo
> que merezca ser recordado para futuras sesiones.

## Contexto del proyecto
(¿Qué es este proyecto? ¿Qué hace? ¿Para quién es?)

## Convenciones
(Estilo de código, patrones de nomenclatura, bibliotecas utilizadas, preferencias de herramientas.)

## Decisiones
(Elecciones que se han tomado y el razonamiento detrás de ellas.)

## Trampas y peculiaridades
(Cosas que hay que recordar sobre este código y que podrían causar problemas
en futuras sesiones: peculiaridades, dependencias no obvias, errores frecuentes.)

## Tareas activas
(En qué se está trabajando actualmente. Elimina las entradas cuando el trabajo se complete.)
"""


# ============================================================
# FUNCIÓN: CARGAR LA MEMORIA DEL AGENTE
# ============================================================

def load_agents_md() -> str:
    """
    Lee el archivo AGENTS.md del workspace, creándolo a partir de la
    plantilla si aún no existe.

    Devuelve:
        str: El contenido completo del archivo como una cadena de texto,
             listo para ser utilizado como contenido de un mensaje de sistema
             en la conversación con el modelo.

    Comportamiento:
        - Si AGENTS.md no existe → se crea con la plantilla por defecto.
        - Si AGENTS.md ya existe → se lee y se devuelve tal cual.
        - Nunca lanza una excepción por archivo inexistente.
    """

    # Paso 1: Garantizar que el archivo existe.
    # Si no existe, se escribe la plantilla inicial. Esta es una
    # restricción rígida que garantiza que el agente nunca se encuentre
    # en un estado de "memoria inexistente". Es análogo a la auto-
    # inicialización del repositorio git en el módulo de herramientas git.
    #
    # Nota: .exists() comprueba si el archivo ya está presente en disco.
    # .write_text() crea el archivo (o lo sobrescribe) con el contenido
    # de la plantilla, codificado en UTF-8 por defecto.
    if not AGENTS_MD_PATH.exists():
        AGENTS_MD_PATH.write_text(AGENTS_MD_TEMPLATE)

    # Paso 2: Leer y devolver el contenido del archivo.
    # El harness (el bucle principal de conversación) se encargará de
    # envolver este texto en un mensaje de sistema con rol "system"
    # antes de enviarlo al modelo, de modo que el agente disponga de
    # toda la memoria acumulada del proyecto desde el primer turno.
    return AGENTS_MD_PATH.read_text()
