import json
import os
import shutil
import urllib.request
import zipfile
from argparse import ArgumentParser

import gradio as gr

from main import song_cover_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')


def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]


def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.Dropdown.update(choices=models_l)


def load_public_models():
    models_table = []
    for model in public_models['voice_models']:
        if not model['name'] in voice_models:
            model = [model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])]
            models_table.append(model)

    tags = list(public_models['tags'].keys())
    return gr.DataFrame.update(value=models_table), gr.CheckboxGroup.update(choices=tags)


def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)

            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise gr.Error(f'В архиве не найден файл .pth. Пожалуйста, проверьте {extraction_folder}.')

    # Переместите файлы модели и индекса в папку извлечения
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    # Удалите все ненужные вложенные папки
    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))


def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Загрузка модели с именем {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Папка модели {dir_name} уже существует! Выберите другое имя для вашей голосовой модели.')

        if 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'

        urllib.request.urlretrieve(url, zip_name)

        progress(0.5, desc='[~] Извлечение архива...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Модель {dir_name} успешно загружена!'

    except Exception as e:
        raise gr.Error(str(e))


def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Папка модели {dir_name} уже существует! Выберите другое имя для вашей голосовой модели.')

        zip_name = zip_path.name
        progress(0.5, desc='[~] Извлечение архива...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Модель {dir_name} успешно загружена!'

    except Exception as e:
        raise gr.Error(str(e))


def filter_models(tags, query):
    models_table = []

    # Без фильтрации
    if len(tags) == 0 and len(query) == 0:
        for model in public_models['voice_models']:
            models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # Фильтрация по тегам и запросу
    elif len(tags) > 0 and len(query) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
                if query.lower() in model_attributes:
                    models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # Фильтрация только по тегам
    elif len(tags) > 0:
        for model in public_models['voice_models']:
            if all(tag in model['tags'] for tag in tags):
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    # Фильтрация только по запросу
    else:
        for model in public_models['voice_models']:
            model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
            if query.lower() in model_attributes:
                models_table.append([model['name'], model['description'], model['credit'], model['url'], model['tags']])

    return gr.DataFrame.update(value=models_table)


def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.Text.update(value=pub_models.loc[event.index[0], 'URL']), gr.Text.update(value=pub_models.loc[event.index[0], 'Model Name'])


def swap_visibility():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)


def process_file_upload(file):
    return file.name, gr.update(value=file.name)


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo == 'mangio-crepe':
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

if __name__ == '__main__':
    parser = ArgumentParser(description='Создание AI кавер-версии песни в директории song_output/id.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Включить возможность обмена")
    parser.add_argument("--listen", action="store_true", default=False, help="Сделать веб-интерфейс доступным из вашей локальной сети.")
    parser.add_argument('--listen-host', type=str, help='Имя хоста, которое будет использоваться сервером.')
    parser.add_argument('--listen-port', type=int, help='Порт, который будет использоваться сервером.')
    args = parser.parse_args()

    # Добавим параметр use_gpu=False, чтобы отключить использование GPU
    web_app = gr.Interface(fn=song_cover_pipeline, inputs="audio", outputs="audio", use_gpu=False)

    voice_models = get_current_models(rvc_models_dir)
    with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
        public_models = json.load(infile)

    with gr.Blocks(title='NoCrypt/miku') as app:

        gr.Label('AISINGERS 🐳 | https://t.me/aisingers', show_label=False)

        # Основная вкладка
        with gr.Tab("Создать"):

            with gr.Accordion('Основные опции'):
                with gr.Row():
                    with gr.Column():
                        rvc_model = gr.Dropdown(voice_models, label='Голосовые модели', info='Папка моделей "AICoverGen --> rvc_models". После добавления новых моделей в эту папку, нажмите кнопку обновления')
                        ref_btn = gr.Button('Обновить модели 🔁', variant='primary')

                    with gr.Column() as yt_link_col:
                        song_input = gr.Text(label='Входная песня', info='Ссылка на песню на YouTube или полный путь к локальному файлу. Для загрузки файла нажмите кнопку ниже.')
                        show_file_upload_button = gr.Button('Загрузить файл')

                    with gr.Column(visible=False) as file_upload_col:
                        local_file = gr.File(label='Аудио файл')
                        song_input_file = gr.UploadButton('Загрузить 📂', file_types=['audio'], variant='primary')
                        show_yt_link_button = gr.Button('Вставить ссылку YouTube/Путь к локальному файлу вместо этого')
                        song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])

                    with gr.Column():
                        pitch = gr.Slider(-3, 3, value=0, step=1, label='Изменение тона (только вокал)', info='Обычно используйте 1 для конверсии мужского в женский голос и -1 наоборот. (Октавы)')
                        pitch_all = gr.Slider(-12, 12, value=0, step=1, label='Общее изменение тона', info='Изменяет тональность/ключ вокала и инструментальной музыки. Небольшие изменения ухудшают качество звука. (Полутон)')
                    show_file_upload_button.click(swap_visibility, outputs=[file_upload_col, yt_link_col, song_input, local_file])
                    show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

            with gr.Accordion('Опции преобразования голоса', open=False):
                with gr.Row():
                    index_rate = gr.Slider(0, 1, value=0.5, label='Коэффициент индексации', info="Управляет сохранением акцента голоса искусственного интеллекта в вокале")
                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Радиус фильтрации', info='Если >=3: применяется медианная фильтрация к результатам извлечения тона. Может снизить дыхание.')
                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='Коэффициент смешивания RMS', info="Управление уровнем подражания громкости оригинального вокала (0) или фиксированным уровнем громкости (1)")
                    protect = gr.Slider(0, 0.5, value=0.33, label='Коэффициент защиты', info='Защита глухих согласных и звуков дыхания. Установите 0.5 для отключения.')
                    with gr.Column():
                        f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='Алгоритм извлечения тона', info='Лучший выбор - rmvpe (четкость в вокале), затем mangio-crepe (более плавный вокал)')
                        crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Длина шага Crepe', info='Меньшие значения приводят к более длительным конверсиям и более высокому риску перебоев в голосе, но обеспечивают лучшую точность тона.')
                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                keep_files = gr.Checkbox(label='Сохранить промежуточные файлы', info='Сохранить все аудиофайлы, созданные в директории song_output/id, например, изолированный вокал/инструментальную музыку. Оставьте без отметки, чтобы сэкономить место')

            with gr.Accordion('Опции аудиосмешивания', open=False):
                gr.Markdown('### Изменение громкости (дБ)')
                with gr.Row():
                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='Основной вокал')
                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Запасной вокал')
                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Музыка')

                gr.Markdown('### Управление реверберацией в искусственном вокале')
                with gr.Row():
                    reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Размер комнаты', info='Чем больше комната, тем дольше реверберация')
                    reverb_wet = gr.Slider(0, 1, value=0.2, label='Уровень влажности', info='Уровень искусственного вокала с реверберацией')
                    reverb_dry = gr.Slider(0, 1, value=0.8, label='Уровень сухости', info='Уровень искусственного вокала без реверберации')
                    reverb_damping = gr.Slider(0, 1, value=0.7, label='Уровень затухания', info='Поглощение высоких частот в реверберации')

                gr.Markdown('### Формат аудиовыхода')
                output_format = gr.Dropdown(['mp3', 'wav'], value='mp3', label='Тип выходного файла', info='mp3: маленький размер файла, приемлемое качество. wav: большой размер файла, лучшее качество')

            with gr.Row():
                clear_btn = gr.ClearButton(value='Очистить', components=[song_input, rvc_model, keep_files, local_file])
                generate_btn = gr.Button("Создать", variant='primary')
                ai_cover = gr.Audio(label='AI Кавер', show_share_button=False)

            ref_btn.click(update_models_list, None, outputs=rvc_model)
            is_webui = gr.Number(value=1, visible=False)
            generate_btn.click(song_cover_pipeline,
                               inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                                       inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                                       protect, pitch_all, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                                       output_format],
                               outputs=[ai_cover])
            clear_btn.click(lambda: [0, 0, 0, 0, 0.5, 3, 0.25, 0.33, 'rmvpe', 128, 0, 0.15, 0.2, 0.8, 0.7, 'mp3', None],
                            outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate,
                                     protect, f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet,
                                     reverb_dry, reverb_damping, output_format, ai_cover])

        # Вкладка для загрузки модели
        with gr.Tab('Загрузить модель'):

            with gr.Tab('С HuggingFace/Pixeldrain URL'):
                with gr.Row():
                    model_zip_link = gr.Text(label='Ссылка для скачивания модели', info='Должен быть zip-файл, содержащий файл модели .pth и опциональный файл .index.')
                    model_name = gr.Text(label='Назовите свою модель', info='Дайте вашей новой модели уникальное имя, отличное от других ваших голосовых моделей.')

                with gr.Row():
                    download_btn = gr.Button('Скачать 🌐', variant='primary', scale=19)
                    dl_output_message = gr.Text(label='Сообщение о выполнении', interactive=False, scale=20)

                download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

                gr.Markdown('## Примеры ввода')
                gr.Examples(
                    [
                        ['https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip', 'Lisa'],
                        ['https://pixeldrain.com/u/3tJmABXA', 'Gura'],
                        ['https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20-%20Sweet%20Magic.zip', 'AZKi'],
                        ['https://pixeldrain.com/u/4v25nMrv', 'Klein']
                    ],
                    label='Примеры',
                    value='Примеры ввода'
                )

            with gr.Tab('С локального устройства'):
                with gr.Row():
                    local_model = gr.File(label='Загрузите zip-файл с вашей моделью', info='Должен быть zip-файл, содержащий файл модели .pth и опциональный файл .index.')
                    model_name_local = gr.Text(label='Назовите свою модель', info='Дайте вашей новой модели уникальное имя, отличное от других ваших голосовых моделей.')

                with gr.Row():
                    upload_btn = gr.Button('Загрузить 📂', variant='primary', scale=19)
                    ul_output_message = gr.Text(label='Сообщение о выполнении', interactive=False, scale=20)

                upload_btn.click(upload_local_model, inputs=[local_model, model_name_local], outputs=ul_output_message)

        # Вкладка для загрузки моделей из интернета
        with gr.Tab("Список публичных моделей"):
            with gr.Row():
                gr.Markdown('## Список публичных голосовых моделей', scale=2)
                with gr.Column() as public_dl_btn_col:
                    public_dl_btn = gr.Button('Загрузить выбранную модель 🌐', variant='primary')
                    public_dl_btn_tip = gr.Text('Выберите модель из списка выше, чтобы увидеть ее параметры и загрузить ее.')
                    gr.download(public_dl_btn, inputs=[public_models['voice_models'], public_dl_btn_tip], outputs=[model_zip_link, model_name])

            with gr.Column():
                with gr.Accordion('Сортировать и фильтровать по'):
                    with gr.Row():
                        filter_query = gr.Text(label='Запрос', info='Поиск по имени, описанию, тегам и кредитам. Регистрозависимо.')
                        filter_query.image('https://huggingface.co/assets/images/transformers_logo.png', width=100, height=30, scale=20)
                    with gr.Row():
                        filter_tags = gr.CheckboxGroup([], label='Теги', info='Отфильтровать модели по тегам. Доступны следующие теги: среднего качества, высокого качества, низкого качества, женский голос, мужской голос, японский, английский, инструментальная музыка, большая модель')
                        filter_tags_tip = gr.Text('Выберите один или несколько тегов для фильтрации моделей по ним.')
                        gr.checkbox(filter_tags, inputs=[public_models['tags'], filter_tags_tip], outputs=[filter_tags])
                    with gr.Row():
                        model_table = gr.DataFrame([], scale=8, width=600, height=300)
                        model_table_tip = gr.Text('Выберите модель из списка выше, чтобы увидеть ее параметры и загрузить ее.')
                        gr.dataframe(model_table, inputs=[public_models['voice_models'], filter_tags, filter_query, public_models['tags'], model_table_tip], outputs=[model_zip_link, model_name])

                with gr.Row():
                    gr.download(public_dl_btn, inputs=[model_zip_link, model_name], outputs=[dl_output_message])
                    public_dl_btn.click(pub_dl_autofill, inputs=[model_table], outputs=[model_zip_link, model_name])

    app.launch(share=args.share_enabled, inbrowser=args.listen, host=args.listen_host, port=args.listen_port)
