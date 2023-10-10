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
        raise gr.Error(f'В архиве не найден файл .pth модели. Пожалуйста, проверьте {extraction_folder}.')

    # Перемещаем модель и файл индекса в папку извлечения
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    # Удаляем ненужные вложенные папки
    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))


def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Загрузка голосовой модели с именем {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Директория голосовой модели {dir_name} уже существует! Выберите другое имя для вашей голосовой модели.')

        if 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'

        urllib.request.urlretrieve(url, zip_name)

        progress(0.5, desc='[~] Извлечение архива...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Голосовая модель {dir_name} успешно загружена!'

    except Exception as e:
        raise gr.Error(str(e))


def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Директория голосовой модели {dir_name} уже существует! Выберите другое имя для вашей голосовой модели.')

        zip_name = zip_path.name
        progress(0.5, desc='[~] Извлечение архива...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Голосовая модель {dir_name} успешно загружена!'

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
    parser = ArgumentParser(description='Генерация AI-покрытия песни в директории song_output/id.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Включить функцию обмена")
    parser.add_argument("--listen", action="store_true", default=False, help="Делать веб-интерфейс доступным из вашей локальной сети.")
    parser.add_argument('--listen-host', type=str, help='Имя хоста, которое будет использоваться сервером.')
    parser.add_argument('--listen-port', type=int, help='Порт прослушивания, который будет использоваться сервером.')
    args = parser.parse_args()

    # Добавим параметр use_gpu=False, чтобы отключить использование GPU
    web_app = gr.Interface(fn=song_cover_pipeline, inputs="audio", outputs="audio", use_gpu=False)

    voice_models = get_current_models(rvc_models_dir)
    with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
        public_models = json.load(infile)

    with gr.Blocks(title='AICoverGenWebUI', theme='NoCrypt/miku'): as app:

        gr.Label('AISINGERS 🐳 | https://t.me/aisingers', show_label=False)

        # Главная вкладка
        with gr.Tab("Генерировать"):

            with gr.Accordion('Основные параметры'):
                with gr.Row():
                    with gr.Column():
                        rvc_model = gr.Dropdown(voice_models, label='Голосовые модели', info='Папка моделей "AICoverGen --> rvc_models". После добавления новых моделей в эту папку, нажмите кнопку обновления')
                        ref_btn = gr.Button('Обновить модели 🔁', variant='primary')

                    with gr.Column() as yt_link_col:
                        song_input = gr.Text(label='Входной трек', info='Ссылка на песню на YouTube или полный путь к локальному файлу. Для загрузки файла нажмите кнопку ниже.')
                        show_file_upload_button = gr.Button('Загрузить файл')

                    with gr.Column(visible=False) as file_upload_col:
                        local_file = gr.File(label='Аудио файл')
                        song_input_file = gr.UploadButton('Загрузить 📂', file_types=['audio'], variant='primary')
                        show_yt_link_button = gr.Button('Вставить ссылку YouTube/Путь к локальному файлу')
                        song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])

                    with gr.Column():
                        pitch = gr.Slider(-3, 3, value=0, step=1, label='Изменение тона (только вокал)', info='Обычно используется 1 для перехода от мужского к женскому вокалу и -1 наоборот. (Октавы)')
                        pitch_all = gr.Slider(-12, 12, value=0, step=1, label='Общее изменение тона', info='Изменяет высоту тона/ключ вокала и инструментальной музыки. Небольшое изменение может ухудшить качество звука. (Полутонов)')
                    show_file_upload_button.click(swap_visibility, outputs=[file_upload_col, yt_link_col, song_input, local_file])
                    show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

            with gr.Accordion('Параметры голосовой конверсии', open=False):
                with gr.Row():
                    index_rate = gr.Slider(0, 1, value=0.5, label='Скорость индекса', info="Контролирует, сколько акцента оставить в голосе искусственного интеллекта")
                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Радиус фильтра', info='Если >=3: применять медианный фильтр к полученным результатам высоты тона. Может уменьшить шумность')
                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='Скорость смешивания RMS', info="Управляет, насколько имитировать громкость исходного вокала (0) или фиксированную громкость (1)")
                    protect = gr.Slider(0, 0.5, value=0.33, label='Скорость защиты', info='Защищает беззвучные согласные и звуки дыхания. Установите 0,5, чтобы отключить.')
                    with gr.Column():
                        f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label='Алгоритм определения тона', info='Лучший выбор - rmvpe (четкость в вокале), затем mangio-crepe (более гладкий вокал)')
                        crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label='Длина шага Crepe', info='Меньшие значения приводят к более длительным конверсиям и большему риску дефицита голоса, но лучшей точности высоты тона.')
                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                keep_files = gr.Checkbox(label='Сохранить промежуточные файлы', info='Сохранить все аудиофайлы, сгенерированные в директории song_output/id, например, изолированный вокал/инструменты. Оставьте недоступным, чтобы сэкономить место')

            with gr.Accordion('Параметры аудио смешивания', open=False):
                gr.Markdown('### Изменение громкости (дБ)')
                with gr.Row():
                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='Основной вокал')
                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Вокал поддержки')
                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Музыка')

                gr.Markdown('### Управление реверберацией на голосе AI')
                with gr.Row():
                    reverb_rm_size = gr.Slider(0, 1, value=0.15, label='Размер комнаты', info='Чем больше комната, тем дольше реверберация')
                    reverb_wet = gr.Slider(0, 1, value=0.2, label='Уровень Wet', info='Уровень звука голоса AI с реверберацией')
                    reverb_dry = gr.Slider(0, 1, value=0.8, label='Уровень Dry', info='Уровень звука голоса AI без реверберации')
                    reverb_damping = gr.Slider(0, 1, value=0.7, label='Уровень затухания', info='Поглощение высоких частот в реверберации')

                gr.Markdown('### Формат аудиовыхода')
                output_format = gr.Dropdown(['mp3', 'wav'], value='mp3', label='Тип выходного файла', info='mp3: маленький размер файла, неплохое качество. wav: большой размер файла, лучшее качество')

            with gr.Row():
                clear_btn = gr.ClearButton(value='Очистить', components=[song_input, rvc_model, keep_files, local_file])
                generate_btn = gr.Button("Генерировать", variant='primary')
                ai_cover = gr.Audio(label='AI Cover', show_share_button=False)

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

            with gr.Tab('По URL HuggingFace/Pixeldrain'):
                with gr.Row():
                    model_zip_link = gr.Text(label='Ссылка для скачивания модели', info='Должен быть файл zip, содержащий файл модели .pth и необязательный файл .index.')
                    model_name = gr.Text(label='Название вашей модели', info='Дайте вашей новой модели уникальное имя от других голосовых моделей.')

                with gr.Row():
                    download_btn = gr.Button('Скачать 🌐', variant='primary', scale=19)
                    dl_output_message = gr.Text(label='Сообщение о выполнении', interactive=False, scale=20)

                download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

                gr.Markdown('## Примеры ввода')
                gr.Examples(
                    [
                        ['https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip', 'Лиза'],
                        ['https://pixeldrain.com/u/3tJmABXA', 'Гура'],
                        ['https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip', 'Ацки']
                    ],
                    [model_zip_link, model_name],
                    [],
                    download_online_model,
                )

            with gr.Tab('Из общего индекса'):

                gr.Markdown('## Как использовать')
                gr.Markdown('- Нажмите "Инициализировать таблицу общедоступных моделей"')
                gr.Markdown('- Фильтруйте модели по тегам или поисковой строке')
                gr.Markdown('- Выберите строку для автозаполнения ссылки для скачивания и названия модели')
                gr.Markdown('- Нажмите "Скачать"')

                with gr.Row():
                    pub_zip_link = gr.Text(label='Ссылка для скачивания модели')
                    pub_model_name = gr.Text(label='Название модели')

                with gr.Row():
                    download_pub_btn = gr.Button('Скачать 🌐', variant='primary', scale=19)
                    pub_dl_output_message = gr.Text(label='Сообщение о выполнении', interactive=False, scale=20)

                filter_tags = gr.CheckboxGroup(value=[], label='Показывать голосовые модели с тегами', choices=[])
                search_query = gr.Text(label='Поиск')
                load_public_models_button = gr.Button(value='Инициализировать таблицу общедоступных моделей', variant='primary')

                public_models_table = gr.DataFrame(value=[], headers=['Название модели', 'Описание', 'Автор', 'URL', 'Теги'], label='Доступные общедоступные модели', interactive=False)
                public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                search_query.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                filter_tags.change(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=pub_dl_output_message)

        # Загрузка вкладки модели
        with gr.Tab('Загрузить локальную модель'):
            with gr.Row():
                local_model_zip = gr.File(label='Zip-архив модели', file_types=['zip'], info='Загрузите архив с моделью')
                local_model_name = gr.Text(label='Название вашей модели', info='Дайте вашей новой модели уникальное имя от других голосовых моделей.')

            with gr.Row():
                upload_btn = gr.Button('Загрузить 📂', variant='primary', scale=19)
                ul_output_message = gr.Text(label='Сообщение о выполнении', interactive=False, scale=20)

            upload_btn.click(upload_local_model, inputs=[local_model_zip, local_model_name], outputs=ul_output_message)

        # О вкладках
        with gr.Tab('О проекте'):

            with gr.Row():
                gr.Text("Привет! Я создал этот проект, чтобы сделать голосовое AI более доступным и интересным.", scale=10)
                gr.Text("Моя цель - предоставить вам возможность создавать интересные и креативные покрытия песен с использованием голосовых моделей, которые вам нравятся.", scale=10)
                gr.Text("Вы можете использовать этот веб-интерфейс для генерации покрытий песен с использованием голосовых моделей и настроек, которые вы выбрали.", scale=10)
                gr.Text("Вы также можете загружать собственные голосовые модели, чтобы сделать процесс более креативным.", scale=10)

            with gr.Row():
                gr.Text("Множество голосовых моделей уже доступно для загрузки из общего индекса, и вы также можете загрузить модели с помощью URL-ссылок.", scale=10)
                gr.Text("Я надеюсь, что вам понравится использовать этот проект, и я с нетерпением жду ваших отзывов и предложений.", scale=10)
                gr.Text("Если у вас есть вопросы или проблемы, пожалуйста, не стесняйтесь обращаться ко мне.", scale=10)

            with gr.Row():
                gr.Text("Вы можете найти меня на GitHub по этой ссылке:", scale=10)
                gr.Link("https://github.com/Varaslav/AICoverGen", "GitHub проекта", scale=10)

            with gr.Row():
                gr.Text("Все голосовые модели по умолчанию включаются в сценарии бесплатно и для общественного использования, благодаря усилиям сообщества и исходным кодам проекта.", scale=10)
                gr.Text("Если вы создаете голосовую модель и хотите добавить ее в общий индекс, отправьте мне сообщение с URL-ссылкой на модель и описанием.", scale=10)

        # Вкладка настроек
        with gr.Tab("Настройки"):

            with gr.Row():
                gr.Text("Настройки могут быть изменены только после перезапуска сервера.", scale=10)

            with gr.Row():
                gr.Text("Дополнительные настройки (например, изменение размера сервера) могут быть заданы в файле config.py.", scale=10)

            with gr.Row():
                gr.Text("Примечание: Некоторые параметры, такие как алгоритм определения тона, могут значительно влиять на скорость и качество обработки.", scale=10)

    # Запустим веб-приложение
    if args.share_enabled:
        web_app.share()
    if args.listen:
        if args.listen_host is not None:
            host = args.listen_host
        else:
            host = "0.0.0.0"
        if args.listen_port is not None:
            port = args.listen_port
        else:
            port = 7860
        web_app.launch(share=False, host=host, port=port)
    else:
        web_app.launch(share=False)
