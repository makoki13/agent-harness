"""
filesystem.py — Herramientas de sistema de ficheros para el agente.

Este módulo define cinco herramientas (read, write, list, mkdir, delete)
que permiten al agente interactuar con un directorio de trabajo acotado
(el «workspace»). Todas las operaciones están confinadas a ese directorio:
ninguna ruta puede escapar de él, lo que actúa como sandbox frente a
accesos accidentales o malintencionados del modelo.

Cada función pública está decorada con @tool, lo que la registra
automáticamente en el registro global de herramientas (registry) al
importarse este módulo. El registro extrae el nombre, el docstring y el
esquema JSON de parámetros de cada función para exponerlos al modelo.

Seguridad:
    La función interna _resolve_path() canonicaliza cualquier ruta
    proporcionada por el modelo y verifica que el resultado sigue dentro
    de WORKSPACE. Esto neutraliza ataques de recorrido de ruta como
    ``../../../etc/passwd``.

Uso:
    No se llama directamente. Se importa por sus efectos secundarios
    desde ``harness/tools/__init__.py``:

        from harness.tools import filesystem  # noqa: F401

    Al ejecutarse el import, los decoradores @tool registran cada función
    en ``registry`` y quedan disponibles para el agente.
"""

from pathlib import Path

from harness.tools.registry import tool

# ---------------------------------------------------------------------------
# Directorio de trabajo (workspace)
# ---------------------------------------------------------------------------
# Todas las herramientas de este módulo operan exclusivamente dentro de
# este directorio. Se resuelve una única vez en el momento del import a
# una ruta absoluta y canónica (sin enlaces simbólicos ni segmentos «..»).
# Esto garantiza que las comprobaciones de seguridad posteriores sean
# fiables y consistentes durante toda la sesión.
WORKSPACE = Path("./workspace").resolve()

# Creamos el directorio de trabajo si no existe. La llamada es idempotente:
# si el directorio ya está creado, no hace nada ni lanza error.
# `exist_ok=True` evita una excepción FileExistsError en ese caso.
WORKSPACE.mkdir(exist_ok=True)


def _resolve_path(path: str) -> Path:
    """
    Resuelve ``path`` contra el workspace y verifica que no escape de él.

    Esta es la barrera de seguridad central del módulo. Toda herramienta
    pública (read, write, list, mkdir, delete) pasa por aquí antes de
    tocar el sistema de ficheros.

    Args:
        path: Ruta relativa proporcionada por el modelo. Se interpreta
              siempre como relativa a WORKSPACE, nunca como absoluta.

    Returns:
        Un objeto Path absoluto y canónico, garantizado dentro de WORKSPACE.

    Raises:
        ValueError: Si la ruta resuelta queda fuera de WORKSPACE. Esto
                    ocurre típicamente con secuencias de «..» que intentan
                    ascender por encima del directorio de trabajo.
    """
    # Paso 1: Unir la ruta proporcionada por el usuario al workspace y
    # canonicalizar el resultado. `.resolve()` expande segmentos «..»,
    # resuelve enlaces simbólicos y convierte la ruta a absoluta.
    # Sin esta canonicalización, una entrada como «../../../etc/passwd»
    # parecería estar dentro del workspace en la unión ingenua, pero
    # realmente apuntaría fuera.
    target = (WORKSPACE / path).resolve()

    # Paso 2: Comprobar que la ruta canónica sigue siendo un descendiente
    # de WORKSPACE. `is_relative_to()` devuelve True solo si `target` está
    # dentro del árbol de directorios de WORKSPACE. Si los «..» sacaron la
    # ruta fuera, esta condición falla y lanzamos ValueError.
    if not target.is_relative_to(WORKSPACE):
        raise ValueError(f"la ruta escapa del workspace: {path}")

    return target


@tool
def read(path: str) -> str:
    """
    Lee el contenido de un fichero del workspace y lo devuelve como texto.

    Args:
        path: Ruta relativa del fichero dentro del workspace.

    Returns:
        El contenido completo del fichero como cadena de texto.

    Raises:
        ValueError: Si la ruta escapa del workspace.
        FileNotFoundError: Si el fichero no existe.
        UnicodeDecodeError: Si el fichero no es texto válido en UTF-8.
    """
    # Resolvemos la ruta de forma segura (verificación de sandbox incluida)
    # y leemos el contenido del fichero en una sola llamada encadenada.
    # `read_text()` abre el fichero, lo lee completo como UTF-8 y lo cierra.
    return _resolve_path(path).read_text()


@tool
def write(path: str, content: str) -> str:
    """
    Escribe contenido en un fichero del workspace.

    Si el fichero no existe, se crea (junto con los directorios padre
    necesarios). Si ya existe, se sobrescribe por completo.

    Args:
        path:    Ruta relativa del fichero dentro del workspace.
        content: Texto que se escribirá en el fichero.

    Returns:
        Mensaje de confirmación con el número de bytes escritos y la ruta.

    Raises:
        ValueError: Si la ruta escapa del workspace.
    """
    # Paso 1: Resolver la ruta de destino de forma segura. Esto valida que
    # la ruta no escape del sandbox antes de realizar ninguna escritura.
    target = _resolve_path(path)

    # Paso 2: Crear los directorios padre si no existen. Esto permite
    # escribir en rutas anidadas como «notes/research/findings.md» sin
    # necesidad de llamar a mkdir previamente. `parents=True` crea toda
    # la cadena de directorios intermedios; `exist_ok=True` evita errores
    # si alguno ya existe.
    target.parent.mkdir(parents=True, exist_ok=True)

    # Paso 3: Escribir el contenido en el fichero. Si el fichero ya
    # existía, se sobrescribe completamente (no se añade al final).
    # `write_text()` codifica como UTF-8 por defecto.
    target.write_text(content)

    # Paso 4: Devolver una confirmación estructurada que el modelo puede
    # verificar. Incluye el número de caracteres escritos y la ruta
    # original (no la canónica) para que coincida con lo que el modelo
    # solicitó.
    return f"escritos {len(content)} bytes en {path}"


@tool
def list(path: str = ".") -> str:
    """
    Lista los ficheros y directorios en una ruta del workspace.

    Args:
        path: Ruta relativa del directorio a listar. Por defecto es «.»,
              es decir, la raíz del workspace.

    Returns:
        Una lista con una entrada por línea. Los directorios se muestran
        con una barra final («/») para distinguirlos de los ficheros.
        Si la ruta no es un directorio, devuelve un mensaje de error.

    Raises:
        ValueError: Si la ruta escapa del workspace.
    """
    # Paso 1: Resolver la ruta de forma segura. Si no se proporciona
    # argumento, se usa «.» que apunta a la raíz del workspace.
    target = _resolve_path(path)

    # Paso 2: Rechazar la operación si el destino no es un directorio.
    # Devolvemos un string de error en lugar de lanzar una excepción para
    # que el modelo pueda leer el mensaje y decidir cómo recuperarse
    # (por ejemplo, corrigiendo la ruta o usando `read` en su lugar).
    if not target.is_dir():
        return f"error: no es un directorio: {path}"

    # Paso 3: Listar las entradas del directorio, ordenarlas alfabéticamente
    # para producir una salida determinista (mismo input → mismo output,
    # útil para tests y para que el modelo no se confunda con cambios de
    # orden entre llamadas). Se añade «/» al final de los directorios para
    # que el modelo pueda distinguirlos de los ficheros de un vistazo.
    entries = sorted(target.iterdir())
    return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries)


@tool
def mkdir(path: str) -> str:
    """
    Crea un directorio en el workspace, incluyendo los directorios padre.

    Si el directorio ya existe, la operación no hace nada ni devuelve error
    (comportamiento idempotente).

    Args:
        path: Ruta relativa del directorio a crear.

    Returns:
        Mensaje de confirmación con la ruta creada.

    Raises:
        ValueError: Si la ruta escapa del workspace.
    """
    # Paso 1: Resolver la ruta de destino de forma segura.
    target = _resolve_path(path)

    # Paso 2: Crear el árbol de directorios completo.
    #   - parents=True: crea todos los directorios intermedios que no
    #     existan (p. ej. para «a/b/c», crea «a», luego «a/b», luego «a/b/c»).
    #   - exist_ok=True: si el directorio ya existe, no lanza
    #     FileExistsError; simplemente no hace nada. Esto hace la
    #     operación idempotente y segura para reintentos.
    target.mkdir(parents=True, exist_ok=True)

    return f"directorio {path} creado"


@tool
def delete(path: str) -> str:
    """
    Elimina un fichero del workspace. No elimina directorios.

    La restricción de no borrar directorios es intencionada: un borrado
    recursivo tiene un radio de impacto demasiado alto para un agente que
    opera directamente sobre el sistema de ficheros del anfitrión. Si se
    necesita eliminar un directorio, se puede hacer a través de la
    herramienta bash (rm -rf), que ofrece un control más explícito.

    Args:
        path: Ruta relativa del fichero a eliminar.

    Returns:
        Mensaje de confirmación con la ruta eliminada, o un mensaje de
        error si se intentó borrar un directorio.

    Raises:
        ValueError: Si la ruta escapa del workspace.
        FileNotFoundError: Si el fichero no existe.
    """
    # Paso 1: Resolver la ruta de forma segura.
    target = _resolve_path(path)

    # Paso 2: Rechazar el borrado de directorios. El borrado recursivo de
    # un árbol de directorios completo es una operación de alto riesgo
    # cuando el workspace reside en el sistema de ficheros del anfitrión.
    # Devolvemos un error legible para que el modelo sepa que la operación
    # fue denegada y pueda informar al usuario o buscar una alternativa.
    if target.is_dir():
        return f"error: no se permite borrar un directorio: {path}"

    # Paso 3: Eliminar el fichero del disco. `unlink()` borra la entrada
    # del sistema de ficheros. Si el fichero no existe, se lanzará
    # FileNotFoundError, que el decorador @tool capturará y convertirá en
    # un string de error legible por el modelo.
    target.unlink()

    return f"eliminado {path}"
