from flask import Flask, request, jsonify
import os
import sys
import vosk
import json
import torch
import numpy as np
import wave
import tempfile
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import array
import struct

app = Flask(__name__)

# Cargar modelo y tokenizador de Hugging Face
model_name = "facebook/bart-large-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model_lenguaje = AutoModelForSequenceClassification.from_pretrained(model_name)


# Función para clasificar un mensaje
def clasificar_mensaje(mensaje, categorias):
    inputs = [f"Este mensaje es sobre {categoria}." for categoria in categorias]
    encoded = tokenizer(
        [mensaje] * len(categorias),
        inputs,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    with torch.no_grad():
        logits = model_lenguaje(**encoded).logits
    entail_contradiction_logits = logits[:, [0, 2]]
    probs = torch.softmax(entail_contradiction_logits, dim=1)[:, 1]
    mejor_categoria = categorias[probs.argmax().item()]
    return mejor_categoria


# Definir las categorías posibles
categorias = [
    "enviar mensaje de texto",
    "enviar audio",
    "hacer llamada",
    "enviar ubicacion",
    "enviar imagen",
    "ver video",
    "Youtube",
]

# Ruta al modelo descargado
MODEL_PATH = "vosk-model-small-es-0.42"


# Descarga del modelo si no existe
def descargar_modelo_vosk():
    import requests
    import zipfile

    if not os.path.exists(MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Descargando modelo Vosk para español...")
        url = "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
        r = requests.get(url, stream=True)
        zip_path = "vosk-model-small-es-0.42.zip"
        with open(zip_path, "wb") as fd:
            for chunk in r.iter_content(chunk_size=128):
                fd.write(chunk)

        print("Descomprimiendo el modelo...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall("./")

        os.remove(zip_path)
        print("Modelo descargado y descomprimido con éxito.")


# Verificar y descargar el modelo
descargar_modelo_vosk()

# Cargar modelo de reconocimiento de voz
model_voz = vosk.Model(MODEL_PATH)


def convertir_audio_segun_necesidad(
    archivo_entrada, archivo_salida, sample_rate_deseado=16000
):
    """
    Convierte un archivo WAV solo cuando sea necesario:
    - De estéreo a mono solo si no es mono
    - A 16 bits solo si no es 16 bits
    - A PCM solo si no es PCM
    También ajusta la frecuencia de muestreo si es necesaria
    """
    with wave.open(archivo_entrada, "rb") as wf:
        # Leer parámetros
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        comp_type = wf.getcomptype()
        comp_name = wf.getcompname()

        # Verificar si es PCM
        es_pcm = comp_type == "NONE" or comp_type == ""

        # Verificar qué conversiones son necesarias
        necesita_mono = n_channels != 1
        necesita_16bits = sample_width != 2
        necesita_pcm = not es_pcm
        necesita_frecuencia = framerate != sample_rate_deseado

        # Si no necesita conversiones
        if not (
            necesita_mono or necesita_16bits or necesita_pcm or necesita_frecuencia
        ):
            # Ya está en el formato correcto
            if archivo_entrada != archivo_salida:
                with open(archivo_entrada, "rb") as f_in:
                    with open(archivo_salida, "wb") as f_out:
                        f_out.write(f_in.read())
            return False, {
                "canales": n_channels,
                "bits_por_muestra": sample_width * 8,
                "frecuencia": framerate,
                "formato": "PCM" if es_pcm else comp_type,
            }

        # Leer todos los frames
        frames = wf.readframes(n_frames)

        # Crear un nuevo archivo WAV con las características deseadas
        with wave.open(archivo_salida, "wb") as wfout:
            # Configurar parámetros de salida según lo necesario
            wfout.setnchannels(1 if necesita_mono else n_channels)
            wfout.setsampwidth(2 if necesita_16bits else sample_width)
            wfout.setframerate(
                sample_rate_deseado if necesita_frecuencia else framerate
            )
            wfout.setcomptype("NONE", "not compressed")  # Siempre PCM para asegurar

            # Aplicar las conversiones necesarias

            # 1. Si necesita conversión de mono/estéreo
            if (
                necesita_mono and n_channels == 2
            ):  # Solo manejar estéreo a mono por ahora
                if sample_width == 1:  # 8 bits
                    # Convertir de estéreo a mono para 8 bits - un byte por muestra
                    mono_frames = bytearray()
                    for i in range(0, len(frames), n_channels):
                        if i < len(frames):
                            # Promedio de los canales
                            mono_val = sum(frames[i : i + n_channels]) // n_channels
                            mono_frames.append(mono_val)
                    frames = bytes(mono_frames)

                elif sample_width == 2:  # 16 bits - dos bytes por muestra
                    # Convertir estéreo a mono para 16 bits
                    mono_frames = bytearray()
                    fmt = "<h"  # little-endian, 16-bit signed int

                    # Manejar las muestras de 2 bytes (16 bits) por canal
                    for i in range(0, len(frames), sample_width * n_channels):
                        if i + (sample_width * n_channels) <= len(frames):
                            # Extraer valores de cada canal
                            canal1 = int.from_bytes(
                                frames[i : i + sample_width],
                                byteorder="little",
                                signed=True,
                            )
                            canal2 = int.from_bytes(
                                frames[i + sample_width : i + 2 * sample_width],
                                byteorder="little",
                                signed=True,
                            )

                            # Calcular promedio
                            mono_val = (canal1 + canal2) // 2

                            # Añadir a los frames mono
                            mono_bytes = mono_val.to_bytes(
                                sample_width, byteorder="little", signed=True
                            )
                            mono_frames.extend(mono_bytes)

                    frames = bytes(mono_frames)

                elif sample_width == 4:  # 32 bits
                    # Manejar las muestras de 4 bytes (32 bits) por canal
                    mono_frames = bytearray()
                    for i in range(0, len(frames), sample_width * n_channels):
                        if i + (sample_width * n_channels) <= len(frames):
                            # Extraer valores de cada canal
                            canal1 = int.from_bytes(
                                frames[i : i + sample_width],
                                byteorder="little",
                                signed=True,
                            )
                            canal2 = int.from_bytes(
                                frames[i + sample_width : i + 2 * sample_width],
                                byteorder="little",
                                signed=True,
                            )

                            # Calcular promedio
                            mono_val = (canal1 + canal2) // 2

                            # Añadir a los frames mono
                            mono_bytes = mono_val.to_bytes(
                                sample_width, byteorder="little", signed=True
                            )
                            mono_frames.extend(mono_bytes)

                    frames = bytes(mono_frames)

            # 2. Si necesita conversión de bits
            if necesita_16bits:
                # Determinar el número de muestras
                num_muestras = len(frames) // sample_width
                if necesita_mono:
                    # Si ya convertimos a mono, el número de canales ya es 1
                    canales_actuales = 1
                else:
                    canales_actuales = n_channels

                if sample_width == 1:  # 8 bits a 16 bits
                    # Convertir 8 bits unsigned a 16 bits signed
                    nuevos_frames = bytearray()
                    for i in range(len(frames)):
                        # Convertir de 8 bits unsigned a 16 bits signed
                        valor_8bit = frames[i]  # 0-255
                        valor_16bit = (valor_8bit - 128) * 256  # Centrar y escalar

                        # Añadir como 16 bits little-endian
                        nuevos_frames.extend(
                            valor_16bit.to_bytes(2, byteorder="little", signed=True)
                        )

                    frames = bytes(nuevos_frames)

                elif sample_width == 3:  # 24 bits a 16 bits
                    nuevos_frames = bytearray()
                    for i in range(0, len(frames), 3):
                        if i + 2 < len(frames):
                            # Leer 24 bits (3 bytes) como little-endian
                            valor_24bit = int.from_bytes(
                                frames[i : i + 3], byteorder="little", signed=True
                            )

                            # Convertir a 16 bits (descartar los 8 bits menos significativos)
                            valor_16bit = valor_24bit >> 8

                            # Añadir como 16 bits little-endian
                            nuevos_frames.extend(
                                valor_16bit.to_bytes(2, byteorder="little", signed=True)
                            )

                    frames = bytes(nuevos_frames)

                elif sample_width == 4:  # 32 bits a 16 bits
                    nuevos_frames = bytearray()
                    for i in range(0, len(frames), 4):
                        if i + 3 < len(frames):
                            # Leer 32 bits (4 bytes) como little-endian
                            valor_32bit = int.from_bytes(
                                frames[i : i + 4], byteorder="little", signed=True
                            )

                            # Convertir a 16 bits (descartar los 16 bits menos significativos)
                            valor_16bit = max(
                                min(valor_32bit >> 16, 32767), -32768
                            )  # Limitar al rango de 16 bits

                            # Añadir como 16 bits little-endian
                            nuevos_frames.extend(
                                valor_16bit.to_bytes(2, byteorder="little", signed=True)
                            )

                    frames = bytes(nuevos_frames)

            # Escribir los frames
            wfout.writeframes(frames)

            # Construir informe de conversión
            conversion_info = {
                "canales_originales": n_channels,
                "canales_nuevos": 1 if necesita_mono else n_channels,
                "bits_por_muestra_original": sample_width * 8,
                "bits_por_muestra_nuevo": 16 if necesita_16bits else sample_width * 8,
                "frecuencia_original": framerate,
                "frecuencia_nueva": (
                    sample_rate_deseado if necesita_frecuencia else framerate
                ),
                "formato_original": comp_type if comp_type else "PCM",
                "formato_nuevo": "PCM",
            }

            # Detallar qué conversiones se hicieron
            conversion_info["conversiones_realizadas"] = {
                "mono": necesita_mono,
                "16bits": necesita_16bits,
                "pcm": necesita_pcm,
                "frecuencia": necesita_frecuencia,
            }

            return True, conversion_info


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "message": "API de procesamiento de audio",
            "endpoint": "/procesar_audio",
            "method": "POST",
            "param": "audio (archivo WAV)",
        }
    )


@app.route("/procesar_audio", methods=["POST"])
def procesar_audio():
    temp_files = []  # Lista para mantener registro de archivos temporales

    try:
        if "audio" not in request.files:
            return jsonify({"error": "No se envió archivo de audio"}), 400

        audio_file = request.files["audio"]
        print(f"📁 Nombre del archivo recibido: {audio_file.filename}")

        # Guardar temporalmente el archivo original
        temp_original = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_files.append(temp_original.name)
        audio_file.save(temp_original.name)
        temp_original.close()

        # Archivo para el audio procesado
        temp_processed = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_files.append(temp_processed.name)
        temp_processed.close()

        # Verificar y convertir el audio solo si es necesario
        try:
            convertido, info_formato = convertir_audio_segun_necesidad(
                temp_original.name, temp_processed.name
            )
            if convertido:
                print(f"✅ Audio convertido exitosamente según necesidad")
                print(f"📊 Información del formato: {info_formato}")
                archivo_procesado = temp_processed.name
            else:
                print(
                    "✅ Audio ya en formato correcto - no se necesitaron conversiones"
                )
                archivo_procesado = temp_original.name
        except Exception as e:
            print(f"❌ Error al preparar el archivo de audio: {str(e)}")
            return (
                jsonify({"error": f"Error al preparar el archivo de audio: {str(e)}"}),
                500,
            )

        # Iniciar el reconocedor
        try:
            wf = wave.open(archivo_procesado, "rb")
            sample_rate = wf.getframerate()

            rec = vosk.KaldiRecognizer(model_voz, sample_rate)
            rec.SetWords(True)

            # Procesar el audio en bloques
            resultado_completo = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break

                if rec.AcceptWaveform(data):
                    resultado = json.loads(rec.Result())
                    if "text" in resultado and resultado["text"].strip():
                        resultado_completo.append(resultado["text"])

            # Obtener el resultado final
            resultado_final = json.loads(rec.FinalResult())
            if "text" in resultado_final and resultado_final["text"].strip():
                resultado_completo.append(resultado_final["text"])

            # Unir todos los fragmentos de texto reconocidos
            mensaje_transcrito = " ".join(resultado_completo)
            if not mensaje_transcrito.strip():
                mensaje_transcrito = resultado_final.get("text", "")

            print(f"📝 Texto reconocido: {mensaje_transcrito}")

            if mensaje_transcrito.strip() == "":
                return (
                    jsonify(
                        {
                            "error": "No se pudo transcribir el audio",
                            "formato_original": (
                                info_formato if "info_formato" in locals() else {}
                            ),
                        }
                    ),
                    422,
                )

            # Clasificar el mensaje
            categoria_detectada = clasificar_mensaje(mensaje_transcrito, categorias)

            # Preparar respuesta con información extra
            respuesta = {
                "texto_transcrito": mensaje_transcrito,
                "categoria": categoria_detectada,
            }

            # Añadir información sobre la conversión si ocurrió
            if "info_formato" in locals():
                respuesta["formato_original"] = info_formato
                if convertido:
                    respuesta["conversion_realizada"] = True

            return jsonify(respuesta)

        except Exception as e:
            print(f"❌ Error al procesar el archivo WAV: {e}")
            return (
                jsonify({"error": f"Error al procesar el archivo de audio: {str(e)}"}),
                500,
            )

    except Exception as e:
        print(f"❌ Error general: {e}")
        return jsonify({"error": f"Error en el servidor: {str(e)}"}), 500
    finally:
        # Limpiar los archivos temporales
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    print(f"🗑️ Archivo temporal eliminado: {temp_file}")
                except Exception as e:
                    print(f"❌ Error al eliminar archivo temporal {temp_file}: {e}")


if __name__ == "__main__":
    # Para desarrollo local
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=False)
