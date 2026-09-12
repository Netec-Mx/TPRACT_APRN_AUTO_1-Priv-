# Predicción de tiempos y costos de atención mediante regresión lineal

## Metadata

| Campo | Detalle |
|---|---|
| **Duración** | 105 minutos |
| **Complejidad** | Media |
| **Nivel de Bloom** | Aplicar / Analizar |
| **Tecnología principal** | Python, Jupyter Notebook y scikit-learn en Visual Studio Code |
| **Modalidad** | Práctica guiada individual |

---

## Descripción general

Durante esta práctica, los participantes aplicarán los conceptos de regresión estudiados en *Supervised Machine Learning: Regression and Classification* sobre un escenario de solicitudes de servicio. A partir de datos históricos, explorarán la relación entre variables operativas, construirán una regresión lineal simple para estimar tiempos de atención y ampliarán el análisis mediante una regresión con múltiples características para estimar costos.

Los participantes ejecutarán y modificarán parámetros definidos, observarán el comportamiento de la función de costo y del descenso del gradiente, aplicarán escalamiento e ingeniería de características y compararán diferentes variantes del modelo mediante métricas y visualizaciones. El foco estará en comprender **qué cambia, por qué cambia y cómo interpretar el resultado**, en lugar de programar los algoritmos desde cero.

---

## Objetivos de aprendizaje

Al completar esta práctica, será capaz de:

- Relacionar una característica con una variable objetivo mediante regresión lineal simple;
- Interpretar pendiente, intercepto, error y calidad de una predicción;
- Observar cómo la tasa de aprendizaje afecta la convergencia del descenso del gradiente;
- Reconocer la vectorización como una forma de operar con múltiples datos y características de manera eficiente;
- Construir una regresión con múltiples características utilizando scikit-learn;
- Aplicar escalamiento e ingeniería de características;
- Comparar un modelo base, un modelo con una característica diseñada y una variante polinómica;
- Generar estimaciones para nuevos casos y explicar las limitaciones de los resultados.

---

## Prerrequisitos

### Conocimientos previos

Se requiere comprensión básica en:

- Regresión lineal;
- Función de costo;
- Descenso del gradiente y tasa de aprendizaje;
- Regresión con múltiples variables;
- Vectorización;
- Escalamiento;
- Ingeniería de características;
- Regresión polinómica.

### Entorno requerido

El equipo de laboratorio debe contar previamente con:

- Visual Studio Code;
- Python 3.11 o superior;
- Extensiones **Python** y **Jupyter** para Visual Studio Code;
- Bibliotecas indicadas en `materiales/requirements.txt`;
- Archivos de la carpeta `materiales`.

La práctica inicia directamente sobre el entorno preparado.

---

## Entorno de laboratorio

### Materiales

Dentro de `Capitulo01/materiales/` encontrará:

1. `01_regresion_tiempos_costos.ipynb` — notebook base de la práctica.
2. `solicitudes_servicio.csv` — datos históricos utilizados en los ejercicios.
3. `requirements.txt` — dependencias requeridas para preparar el entorno.

### Datos de trabajo

| Campo | Descripción |
|---|---|
| `solicitud_id` | Identificador de la solicitud |
| `usuarios_afectados` | Cantidad de usuarios asociados al caso |
| `incidencias_reportadas` | Número de incidencias relacionadas |
| `complejidad_servicio` | Nivel de complejidad, de 1 a 5 |
| `horas_estimadas` | Esfuerzo estimado antes de la atención |
| `cambios_recientes` | Indica si existen cambios recientes relacionados (0/1) |
| `canales_involucrados` | Cantidad de canales o componentes involucrados |
| `tiempo_atencion_horas` | Tiempo real de atención |
| `costo_atencion_usd` | Costo final registrado |

---

## Procedimiento paso a paso

### Preparación — 5 minutos

1. Abra **Visual Studio Code** y seleccione **File > Open Folder**. Abra `Capitulo01/materiales`.
2. Abra `01_regresion_tiempos_costos.ipynb`.
3. Seleccione **Select Kernel > Python Environments** y elija **Python 3.11 o 3.12**.
4. Ejecute la primera celda del notebook para instalar las dependencias:

   ```python
   %pip install -r requirements.txt --progress-bar on
   ```

5. Si Visual Studio Code solicita reiniciar el kernel, seleccione **Restart**.
6. Ejecute la siguiente celda de preparación y confirme que carga `solicitudes_servicio.csv` con **600 registros**.

> **Nota:** si Python no está instalado, descárguelo desde `https://www.python.org/downloads/`, instale Python 3.11 o 3.12 de 64 bits y active **Add Python to PATH** durante la instalación. Después cierre y vuelva a abrir Visual Studio Code.

**Resultado esperado:** dependencias instaladas, kernel activo y 600 registros disponibles.

**Verificación:**
- [ ] El notebook ejecuta celdas sin errores de importación.
- [ ] Se muestran 600 registros.

---

### Reto 1 — Comprender el problema a partir de los datos — 10 minutos

**Objetivo:** identificar las variables disponibles y formular correctamente los problemas de regresión que se resolverán.

1. Ejecute las celdas de **1. Explorar el problema y los datos**.
2. Revise las primeras filas y las estadísticas descriptivas.
3. Observe el gráfico **Incidencias reportadas vs. tiempo de atención**.

**Resultado esperado:** identificación de `tiempo_atencion_horas` como objetivo del modelo simple y reconocimiento de que existen otras variables que pueden afectar el resultado.

---

### Reto 2 — Construir e interpretar una regresión lineal simple — 15 minutos

**Objetivo:** relacionar el número de incidencias con el tiempo de atención e interpretar el modelo obtenido.

1. Ejecute las celdas de **2. Regresión lineal simple**.
2. Observe el intercepto y la pendiente.
3. Interprete la pendiente con sus propias palabras.
4. Revise las métricas `MAE`, `RMSE` y `R²`.
5. Observe la recta sobre los datos de entrenamiento.

**Resultado esperado:** una interpretación del modelo en lenguaje operativo, diferenciando asociación de causalidad.

---

### Reto 3 — Experimentar con función de costo, descenso del gradiente y tasa de aprendizaje — 15 minutos

**Objetivo:** observar cómo la tasa de aprendizaje modifica la velocidad y estabilidad de la convergencia.

El notebook contiene una implementación preparada de la función de costo y del descenso del gradiente. En este reto se modifica únicamente el parámetro de experimentación.

1. Ejecute la sección **3. Función de costo y descenso del gradiente**.
2. Observe las curvas para:
   - `alpha = 0.01`;
   - `alpha = 0.10`;
   - `alpha = 2.10`.
3. Compare el costo inicial y final de cada ejecución.
4. Identifique:
   - cuál converge lentamente;
   - cuál converge de forma estable;
   - cuál provoca crecimiento del costo.
5. Cambie temporalmente `ALPHAS` por otro valor intermedio, por ejemplo `0.30`, y vuelva a ejecutar.

**Punto de aprendizaje:** el valor de `alpha` controla el tamaño de cada actualización. Un valor útil permite que el costo disminuya de manera estable hacia un mínimo.

---

### Reto 4 — Ampliar el modelo con múltiples características — 20 minutos

**Objetivo:** estimar el costo de atención utilizando información operativa adicional y observar el papel del escalamiento y la vectorización.

1. Ejecute **4. Regresión con múltiples características y escalamiento**.
2. Revise `CARACTERISTICAS_BASE` y confirme qué información recibe el modelo.
3. Observe las métricas del modelo multivariable.
4. Revise los coeficientes estandarizados y localice las características con mayor magnitud.
5. Ejecute la celda de **vectorización**.
6. Confirme que la predicción calculada mediante `X @ w + b` coincide con la salida del pipeline.

**Resultado esperado:** un modelo de costo con varias características y comprensión del uso de `StandardScaler`, coeficientes y operación vectorizada.

---

### Reto 5 — Probar ingeniería de características y una variante polinómica — 20 minutos

**Objetivo:** comprobar si una representación diferente de los datos mejora la capacidad predictiva.

1. Ejecute **5. Ingeniería de características y regresión polinómica**.
2. Revise la nueva característica:

```text
carga_operativa = incidencias_reportadas × complejidad_servicio
```

3. Compare los tres modelos mostrados:
   - modelo base;
   - modelo con `carga_operativa`;
   - modelo polinómico de grado 2.
4. Compare principalmente `MAE`, `RMSE` y `R²` sobre el conjunto de prueba.
5. Observe el gráfico de residuos del modelo con `carga_operativa`.

**Punto de aprendizaje:** agregar complejidad al modelo tiene valor cuando mejora su comportamiento sobre datos de prueba y existe una interpretación razonable de la nueva representación.

---

### Reto 6 — Generar estimaciones sobre nuevos casos — 15 minutos

**Objetivo:** utilizar el modelo seleccionado para analizar solicitudes nuevas e interpretar sus resultados antes de utilizarlos en una decisión.

1. Ejecute **6. Predecir nuevos casos e interpretar el modelo**.
2. Compare los tres escenarios propuestos.
3. Identifique cómo cambia el costo estimado entre un caso de baja, media y alta carga operativa.
4. Revise los coeficientes estandarizados del modelo.
5. Seleccione uno de los casos y explique qué características contribuyen a que su estimación sea mayor o menor.

#### Apoyo opcional con Copilot Chat Standard

Puede copiar únicamente las métricas y coeficientes mostrados por el notebook y utilizar el siguiente prompt:

```text
Estoy aprendiendo regresión lineal y obtuve estos resultados de un modelo de costo:

[PEGAR MÉTRICAS Y COEFICIENTES]

Ayúdame a interpretarlos en lenguaje sencillo.
Diferencia claramente:
- qué puedo afirmar a partir del modelo;
- qué sería solo una asociación;
- qué conclusiones requerirían información adicional.

Después hazme dos preguntas para comprobar si entendí la interpretación.
```

Revise la explicación y contraste cualquier afirmación con los resultados del notebook.

**Resultado esperado:** predicciones para nuevos casos acompañadas de una interpretación técnica y operativa.

---

### Cierre — 5 minutos

Complete una última celda Markdown:

```markdown
## Conclusiones de la práctica

Conecta los resultados técnicos con el caso de uso y responde:

1. ¿Qué diferencia observaste entre utilizar una sola característica y utilizar varias?
2. ¿Qué ocurrió cuando modificaste la tasa de aprendizaje?
3. ¿Qué aportó la característica `carga_operativa`?
4. ¿Qué modelo elegirías para estimar costo y qué evidencia respalda tu decisión?
5. ¿Qué revisión humana mantendrías antes de utilizar una predicción para planificación operativa?
```

---

## Validación y pruebas finales

La práctica se considera completada cuando:

- [ ] el notebook carga correctamente los 600 registros;
- [ ] se ejecutó la regresión lineal simple;
- [ ] se interpretaron pendiente, MAE, RMSE y R²;
- [ ] se compararon al menos tres tasas de aprendizaje;
- [ ] se observó el efecto del descenso del gradiente sobre la función de costo;
- [ ] se ejecutó la regresión multivariable con escalamiento;
- [ ] se comprobó una operación vectorizada;
- [ ] se compararon el modelo base, la característica `carga_operativa` y la variante polinómica;
- [ ] se generaron predicciones para nuevos casos;
- [ ] se justificó la selección final del modelo utilizando métricas de prueba.

---

## Solución de problemas

### El notebook solicita seleccionar un kernel

1. Seleccione **Select Kernel** en la parte superior derecha.
2. Elija **Python Environments**.
3. Seleccione el entorno preparado para el curso.
4. Ejecute nuevamente la primera celda.

### Aparece `ModuleNotFoundError`

Seleccione el kernel correspondiente al entorno preparado para el curso. Las dependencias requeridas están documentadas en `requirements.txt`.

### No se encuentra `solicitudes_servicio.csv`

Abra `Capitulo01/materiales` como carpeta de trabajo en Visual Studio Code y vuelva a ejecutar la primera celda.

### La curva de costo crece rápidamente

Revise el valor utilizado en `ALPHAS`. Compare el comportamiento con `0.01` y `0.10` y relacione el resultado con el tamaño de los pasos del descenso del gradiente.

---

## Limpieza del entorno

1. Guarde el notebook mediante **Ctrl+S**.
2. Cierre el notebook y Visual Studio Code cuando finalice la sesión.
3. Conserve el archivo trabajado para utilizarlo como referencia durante las siguientes prácticas.

---

## Resumen

Durante esta práctica se aplicaron los principales conceptos de regresión estudiados en las primeras dos semanas del curso. Se comenzó con una regresión lineal simple, se observó el comportamiento de la función de costo y el descenso del gradiente y se experimentó con diferentes tasas de aprendizaje. Posteriormente, se incorporaron múltiples características, escalamiento, vectorización e ingeniería de características para estimar costos de atención.

La comparación entre modelos permitió comprobar que la calidad de una solución debe evaluarse mediante resultados sobre datos de prueba y no únicamente por su complejidad. Finalmente, se generaron predicciones para nuevos casos y se interpretaron los resultados considerando que los coeficientes representan asociaciones aprendidas a partir de los datos.

---

## Recursos adicionales

- Coursera — Supervised Machine Learning: Regression and Classification: https://www.coursera.org/learn/machine-learning
- scikit-learn — Linear models: https://scikit-learn.org/stable/modules/linear_model.html
- scikit-learn — StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
- scikit-learn — PolynomialFeatures: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
- Visual Studio Code — Python and Jupyter notebooks: https://code.visualstudio.com/docs/languages/python
