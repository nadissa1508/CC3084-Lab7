# Codebook — Laboratorio 7 (Spark MLlib)

Detalle de las variables analíticas utilizadas, derivadas de la base de
Personas de la ENEIC (INE Guatemala).

## Variables originales -> analíticas

| Variable original | Nombre analítico | Tipo | Uso | Descripción |
|---|---|---|---|---|
| `P05D01` | `salario_mensual` | numérica (Q) | Variable objetivo | Sueldo o salario mensual sin descuentos de la ocupación principal |
| `P02A03` | `edad` | numérica | Predictor / clustering | Edad de la persona |
| `P05C07A` | `antiguedad_anios_raw` | numérica | Insumo de `antiguedad` | Años de antigüedad reportados |
| `P05C07B` | `antiguedad_meses_raw` | numérica | Insumo de `antiguedad` | Meses adicionales de antigüedad (0-11) |
| — | `antiguedad` | numérica (derivada) | Predictor / clustering | `antiguedad_anios_raw + antiguedad_meses_raw/12` |
| `P05H01A` | `horas_semanales` | numérica | Predictor / clustering | Horas habituales semanales en la ocupación principal |
| `P03A03A` | `nivel_educativo` | categórica | Predictor | Nivel educativo alcanzado. **Código 0 = "ninguno" (no es faltante)** |
| `P05C16` | `categoria_ocupacional` | categórica | Filtro y predictor | 1=Empleado de gobierno, 2=Empleado de empresa privada, 3=Empleado jornalero o peón, 4=Servicio doméstico |
| `DOMINIO` | `dominio` | categórica | Predictor | Dominio de muestreo / región |
| `OCUPADOS` | `ocupado` | numérica (flag) | Filtro | 1 = persona ocupada |
| `NUM_HOGAR` | `NUM_HOGAR` | identificador | Auditoría | Número de hogar |
| `NUM_PERSONA` | `NUM_PERSONA` | identificador | Auditoría | Número de persona dentro del hogar |
| `FACTOR` | `FACTOR` | numérica | Documentación diseño muestral | Factor de expansión (no se usa para ponderar en este laboratorio) |
| `ANIO` | `ANIO` | numérica | Auditoría de fuente | Año reportado en el archivo original |
| `TRIMESTRE` | `TRIMESTRE` | numérica | Auditoría de fuente | Código de trimestre tal como viene en el archivo (no usar directamente como trimestre calendario) |

## Variables de trazabilidad (creadas en `src/io_utils.py`)

| Variable | Descripción |
|---|---|
| `archivo_origen` | Nombre del archivo crudo del que proviene el registro |
| `periodo_archivo` | Identificador del corte publicado, ej. `2025T1` |
| `anio_archivo` | Año del archivo de procedencia |
| `trimestre_calendario` | Trimestre calendario del archivo de procedencia (1-4) |

## Población analítica (filtros aplicados en orden, ver `src/prepare.py::FILTER_STEPS`)

1. `ocupado == 1`
2. `categoria_ocupacional` en {1, 2, 3, 4}
3. `salario_mensual` numérico, finito y > 0
4. `edad` numérica, finita y >= 15
5. `antiguedad_anios_raw >= 0`
6. `antiguedad_meses_raw` entero entre 0 y 11
7. `antiguedad <= edad`
8. `0 < horas_semanales <= 168`

## Predictores del modelo supervisado (exactamente 6)

`edad`, `antiguedad`, `horas_semanales`, `nivel_educativo`, `categoria_ocupacional`, `dominio`.

## Códigos válidos de variables categóricas

Tomados de las etiquetas de valor embebidas en los `.sav` (verificado que
son idénticos en las 5 bases con `pyreadstat`), y confirmados contra los
diccionarios de datos en `data/raw/`.

**`nivel_educativo` (`P03A03A`)**

| Código | Etiqueta |
|---|---|
| 0 | NINGUNO |
| 1 | PREPRIMARIA |
| 2 | PRIMARIA |
| 3 | BASICO |
| 4 | DIVERSIFICADO |
| 5 | SUPERIOR |
| 6 | MAESTRIA |
| 7 | DOCTORADO |

**`dominio` (`DOMINIO`)**

| Código | Etiqueta |
|---|---|
| 1 | Urbano Metropolitano |
| 2 | Resto Urbano |
| 3 | Rural Nacional |

**`categoria_ocupacional` (`P05C16`)** — universo completo tiene 9 códigos
(incluye cuenta propia, patronos, no remunerados); este laboratorio filtra
únicamente a asalariados, códigos 1-4 (ver tabla en la sección de arriba).

Cualquier código fuera de estas listas, o valor nulo, se homologa a
`DESCONOCIDO` en `src/prepare.py::apply_categorical_validation`.
