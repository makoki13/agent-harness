"""
Herramientas de Git: encapsulan operaciones de commit/diff/log/checkout/branch
como herramientas del harness (entorno de ejecución del agente).

Todas las operaciones se ejecutan con el directorio de trabajo (workspace)
como directorio activo — esto está fijado en el código de forma rígida,
no lo proporciona el modelo. El workspace se inicializa automáticamente como
un repositorio git en el momento de la importación del módulo, de modo que
el agente nunca tenga que recordar ejecutar `git init`.
"""

# ============================================================
# IMPORTACIONES
# ============================================================

# subprocess: permite ejecutar comandos externos (en este caso, git)
# como subprocesos del sistema, capturando su salida y código de retorno.
import subprocess

# WORKSPACE: constante (objeto Path) que apunta al directorio de trabajo
# del agente. Se importa desde el módulo de herramientas de sistema de archivos.
from harness.tools.filesystem import WORKSPACE

# tool: decorador del registro de herramientas del harness.
# Al aplicar @tool a una función, esta queda registrada y disponible
# para que el agente la invoque durante la conversación.
from harness.tools.registry import tool

# ============================================================
# CONFIGURACIÓN GLOBAL
# ============================================================

# Tiempo máximo de espera (timeout) para cada invocación de git, en segundos.
# Las operaciones de git en un repositorio local deberían ser casi instantáneas.
# Si una operación se queda colgada (por ejemplo, esperando entrada interactiva),
# se mata el proceso en lugar de congelar al agente indefinidamente.
GIT_TIMEOUT = 10


# ============================================================
# FUNCIÓN INTERNA: EJECUCIÓN DE COMANDOS GIT
# ============================================================

def _run_git(*args: str) -> str:
    """
    Ejecuta un comando git dentro del workspace y devuelve la salida combinada
    como una cadena de texto.

    El directorio de trabajo del subproceso está fijado de forma rígida a
    WORKSPACE — el modelo no puede influir en él ni modificarlo.
    Los códigos de salida distintos de cero NO lanzan una excepción;
    en su lugar, la salida se devuelve con una nota que incluye el código
    de salida, para que el modelo pueda leerla y decidir qué hacer a continuación.
    """

    # Paso 1: Invocar git usando la forma de lista de argumentos (sin shell).
    # Al NO usar shell=True, se elimina el riesgo de inyección de comandos:
    # cada argumento se pasa de forma literal al ejecutable git.
    # cwd=WORKSPACE es la restricción rígida que limita el alcance de la operación
    # al directorio de trabajo del agente.
    result = subprocess.run(
        ["git", *args],       # Comando: git seguido de los argumentos recibidos
        cwd=WORKSPACE,        # Directorio de trabajo fijo (el workspace del agente)
        capture_output=True,  # Capturar stdout y stderr en lugar de imprimirlos
        text=True,            # Devolver la salida como str (no como bytes)
        timeout=GIT_TIMEOUT,  # Matar el proceso si tarda más de GIT_TIMEOUT segundos
        check=False,          # No lanzar excepción si el código de salida != 0;
                              # el modelo se encargará de interpretar el error.
    )

    # Paso 2: Extraer el texto de salida del comando.
    # Se prefiere stdout; solo se recurre a stderr cuando stdout está vacío.
    # Esto se debe a que algunos comandos de git (como `log` en un repositorio
    # vacío) escriben información útil en stderr en lugar de stdout.
    output = result.stdout.strip() or result.stderr.strip()

    # Paso 3: Si git devolvió un código de salida distinto de cero (error),
    # se antepone una nota con el código de salida para que el modelo pueda
    # reconocer el error y recuperarse sin necesidad de interpretar códigos
    # de retorno numéricos por sí mismo.
    if result.returncode != 0:
        return f"[git exit {result.returncode}] {output}"

    # Si todo fue correcto pero la salida está vacía, devolvemos un marcador
    # explícito para que el modelo sepa que el comando se ejecutó sin producir
    # texto visible.
    return output or "(sin salida)"


# ============================================================
# INICIALIZACIÓN AUTOMÁTICA DEL REPOSITORIO GIT
# ============================================================
# Este bloque se ejecuta una única vez, en el momento de importar el módulo.
# Comprueba si ya existe un directorio .git dentro del workspace.
# Si no existe, inicializa el repositorio y configura valores por defecto.
# Es una restricción rígida que previene fallos del tipo "olvidé hacer git init".
if not (WORKSPACE / ".git").exists():
    # Inicializar un repositorio git vacío en el workspace.
    _run_git("init")

    # Establecer un nombre de rama por defecto ("main") y una identidad
    # de usuario básica para que los commits funcionen directamente,
    # incluso en sistemas que no tengan configuración global de git.
    _run_git("config", "user.name", "agent")
    _run_git("config", "user.email", "agent@harness.local")

    # Apuntar HEAD a refs/heads/main (en lugar de master u otro nombre).
    # symbolic-ref modifica la referencia simbólica de HEAD sin crear el commit,
    # lo cual es necesario porque el repositorio aún no tiene ningún commit.
    _run_git("symbolic-ref", "HEAD", "refs/heads/main")


# ============================================================
# HERRAMIENTA: git_status
# ============================================================

@tool
def git_status() -> str:
    """
    Muestra el estado actual del árbol de trabajo:
    archivos modificados, preparados (staged) y no rastreados (untracked).

    Usa la bandera --short para obtener una salida compacta y legible
    por el modelo, en lugar del formato extenso por defecto de git status.
    """
    return _run_git("status", "--short")


# ============================================================
# HERRAMIENTA: git_diff
# ============================================================

@tool
def git_diff(path: str = "") -> str:
    """
    Muestra los cambios no preparados (unstaged) en el workspace.
    Opcionalmente, se puede restringir la salida a una única ruta (archivo o directorio).

    Parámetros:
        path (str): Ruta relativa dentro del workspace. Si se omite o está vacía,
                    se muestran los cambios de todos los archivos.
    """
    # Paso 1: Construir la lista de argumentos.
    # Si path está vacío, se muestra el diff de todo el repositorio.
    # Si se proporciona una ruta, se añade como filtro.
    args = ["diff"]
    if path:
        args.append(path)

    # Paso 2: Ejecutar git diff con los argumentos construidos.
    return _run_git(*args)


# ============================================================
# HERRAMIENTA: git_log
# ============================================================

@tool
def git_log(limit: int = 10) -> str:
    """
    Muestra el historial reciente de commits.
    Por defecto, devuelve los últimos 10 commits.

    Parámetros:
        limit (int): Número máximo de commits a mostrar.

    Se usa --oneline para mantener la salida compacta (un commit por línea,
    con hash abreviado y mensaje). Si el modelo necesita más detalle sobre
    un commit concreto, puede leerlo posteriormente con otras herramientas.
    """
    return _run_git("log", f"--max-count={limit}", "--oneline")


# ============================================================
# HERRAMIENTA: git_commit
# ============================================================

@tool
def git_commit(message: str) -> str:
    """
    Prepara (stage) todos los cambios actuales y los confirma (commit)
    con el mensaje proporcionado.

    Devuelve el hash del commit si la operación tiene éxito,
    o un mensaje de error si falla (por ejemplo, si no hay nada que confirmar).

    Parámetros:
        message (str): Mensaje descriptivo del commit.

    Flujo interno:
        1. git add -A  → prepara todos los cambios (nuevos, modificados, eliminados)
        2. git commit -m "mensaje" → crea el commit

    Nota: un código de salida != 0 es habitual aquí (por ejemplo, cuando no hay
    cambios que confirmar). _run_git devuelve la explicación al modelo en lugar
    de lanzar una excepción, para que este pueda decidir cómo proceder.
    """
    # Paso 1: Preparar todos los cambios actuales.
    # Equivalente a ejecutar `git add -A` en la terminal.
    # -A añade archivos nuevos, modificados y eliminados al área de preparación.
    _run_git("add", "-A")

    # Paso 2: Crear el commit con el mensaje indicado.
    # Si no hay cambios preparados, git devolverá un código de salida != 0
    # y _run_git incluirá la nota "[git exit 1] ..." en la respuesta.
    return _run_git("commit", "-m", message)


# ============================================================
# HERRAMIENTA: git_checkout
# ============================================================

@tool
def git_checkout(ref: str) -> str:
    """
    Cambia a un commit o a una rama.

    Se utiliza tanto para retroceder (rollback) a un commit anterior
    como para cambiar a una rama hermana.

    Parámetros:
        ref (str): Referencia de destino. Puede ser:
                   - Un hash de commit (obtenido previamente con git_log)
                   - Un nombre de rama (obtenido previamente con git_branch)

    Nota de uso típico:
        El agente suele llamar primero a git_log para encontrar el hash del
        commit al que quiere retroceder, o a git_branch para encontrar el nombre
        de la rama a la que quiere cambiar, y luego invoca git_checkout con esa referencia.
    """
    return _run_git("checkout", ref)


# ============================================================
# HERRAMIENTA: git_branch
# ============================================================

@tool
def git_branch(name: str = "") -> str:
    """
    Lista las ramas existentes si se llama sin argumento.
    Crea una nueva rama si se proporciona un nombre.

    IMPORTANTE: esta herramienta NO cambia a la nueva rama.
    Para cambiar de rama, hay que llamar a git_checkout(name) a continuación.

    Parámetros:
        name (str): Nombre de la nueva rama a crear.
                    Si está vacío, se listan todas las ramas.

    Flujo típico para experimentar en una rama nueva:
        1. git_branch("experimento")  → crea la rama
        2. git_checkout("experimento") → cambia a ella
    """
    # Paso 1: Sin nombre → listar ramas.
    # La salida muestra todas las ramas locales, con un asterisco (*)
    # junto a la rama actualmente activa.
    if not name:
        return _run_git("branch")

    # Paso 2: Con nombre → crear una nueva rama en el HEAD actual.
    # La rama se crea apuntando al mismo commit que HEAD, pero NO se
    # cambia a ella. El agente debe llamar git_checkout(name) por separado
    # para realizar el cambio efectivo.
    return _run_git("branch", name)
