import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

# --- ГЛОБАЛЬНАЯ КОНФИГУРАЦИЯ ---
DATA_CONFIG = {
    # Параметры сетки
    'nz': 32,
    'nx': 128,
    'ny': 128,
    
    # Физические пороги и константы
    'building_threshold': -9000,  # Значение в .plt, означающее твердое тело
    'wind_eps': 1e-9,             # Малое число для предотвращения деления на 0
    
    # Слои для таргета (Z-индексы)
    'target_z_indices': [1, 5, 8, 25],
    
    # Паттерны файлов
    'conc_file_pattern': r'C\[\d+\]-avg-\.plt',
    'tracer_id_regex': r'C\[(\d+)\]',
    
    # Пути
    'raw_root': '/app/urban-layer-datasets/',
    'save_root': '/app/urban-layer-datasets/2026_01_19_500_25d_data_2' # '/app/urban-layer-datasets/2026_01_19_500_25d_data' , /app/urban-layer-datasets/2026_01_27_500_25d_data'
}

# --- НАСТРОЙКИ ---
# RAW_ROOT = '/app/urban-layer-datasets/'
# SAVE_ROOT = '/app/urban-layer-datasets/2026_01_19_500_25d_data_1'  
# GRID_DIMS = {'z': 32, 'y': 128, 'x': 128}

# Целевые слои для декодера
# TARGET_Z_INDICES = [1, 5, 8, 25] 

# Словарь для маппинга осей Tecplot в индексы Numpy (Z, Y, X)
# В файлах порядок X, Y, Z. Numpy reshape делает (Z, Y, X).
# X меняется быстрее всего (внутренний цикл), Z медленнее всего.


def extract_number(filename):
    match = re.search(DATA_CONFIG['tracer_id_regex'], filename)
    return int(match.group(1)) if match else float('inf') 

def sort_filenames(filenames):
    return sorted(filenames, key=extract_number)

def get_filenames(file_path, file_pattern):
    file_list = [f for f in os.listdir(file_path) 
                 if os.path.isfile(os.path.join(file_path, f)) and re.search(file_pattern, f)]
    return sort_filenames(file_list)


class RobustConfigParser:
    """Парсер конфига (тот же, что и раньше, сокращен для краткости)"""
    def __init__(self, filepath):
        self.filepath = filepath
        with open(filepath, 'r') as f:
            self.lines = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        self.domain_params = self._parse_simple_block('domain')
        self.grid_params = self._parse_simple_block('grid')
        self.wind_params = self._extract_wind_forcing()
        self.tracers = self._parse_tracers_robust()

        len_x = self.domain_params.get('length', 256.0)
        len_y = self.domain_params.get('width', 256.0)
        len_z = self.domain_params.get('height', 64.0)
        
        self.scale_x = self.grid_params.get('cx', DATA_CONFIG['nx']) / len_x
        self.scale_y = self.grid_params.get('cy', DATA_CONFIG['ny']) / len_y
        self.scale_z = self.grid_params.get('cz', DATA_CONFIG['nz'])  / len_z
        self.max_z_idx = self.grid_params.get('cz', DATA_CONFIG['nz'])

    def _extract_value(self, text):
        match = re.search(r'=\s*([-\d.eE]+)', text)
        return float(match.group(1)) if match else 0.0

    def _extract_value_for_emission(self, text, key):
        """
        Пытается извлечь число для конкретного ключа.
        Ищет паттерн: граница_слова + ключ + пробелы + равно + число
        """
        # Ищем конкретный ключ, за которым следует равно и число.
        # \b означает границу слова (чтобы поиск 'min' не нашел 'xmin')
        pattern = rf'\b{key}\s*=\s*([-\d.eE]+)'
        
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
        return None  # Лучше возвращать None, если не найдено, чем 0.0
    
    def _extract_wind_forcing(self):
        wind = {'dPdx': 0.0, 'dPdy': 0.0}
        for line in self.lines:
            if 'dPdx' in line: wind['dPdx'] = self._extract_value(line)
            if 'dPdy' in line: wind['dPdy'] = self._extract_value(line)
        return wind

    def _parse_simple_block(self, block_name):
        params = {}
        in_block = False
        for line in self.lines:
            if line.startswith(block_name) and '{' in line: in_block = True; continue
            if in_block:
                if '}' in line: break
                parts = line.split(';')
                for part in parts:
                    if '=' in part:
                        k, v = part.split('=')
                        try: params[k.strip()] = float(re.search(r'([-\d.eE]+)', v).group(1))
                        except: pass
        return params

    def _parse_tracers_robust(self):
        tracers = {}
        current_tracer = None
        current_content = []
        brace_count = 0
        in_tracer_block = False
        for line in self.lines:
            match_start = re.match(r'tracer_(\d+)\s*\{', line)
            if match_start:
                current_tracer = int(match_start.group(1))
                in_tracer_block = True; brace_count = 1; current_content = []; continue
            if in_tracer_block:
                brace_count += line.count('{') - line.count('}')
                current_content.append(line)
                if brace_count == 0:
                    tracers[current_tracer] = self._extract_emissions(current_content)
                    in_tracer_block = False
        return tracers

    def _extract_emissions(self, lines):
        sources = []
        in_emission = False
        current_source = {}
        for line in lines:
            if 'point_emission_' in line: in_emission = True; current_source = {}; continue
            
            if in_emission:
                if '}' in line:
                    if current_source: sources.append(current_source)
                    in_emission = False
                    continue
                
                # Парсим координаты
                for axis in ['xmin', 'xmax', 'ymin', 'ymax']:
                    # Передаем axis в функцию
                    val = self._extract_value_for_emission(line, axis)
                    
                    # Если значение найдено, записываем его
                    if val is not None:
                        current_source[axis] = val
                current_source['zmin'] = 0.0
                current_source['zmax'] = 8.0
        
        return sources
    
    def get_source_height_map(self, tracer_id, shape=(DATA_CONFIG['ny'], DATA_CONFIG['nx'])):
        """
        Генерирует 2D карту (Y, X), где значение пикселя = нормированная высота источника (0..1).
        Если источника нет - 0.
        """
        h_map = np.zeros(shape, dtype=np.float32)
        sources = self.tracers.get(tracer_id, [])
        for box in sources:
            try:
                x1 = int(box.get('xmin', 0) * self.scale_x)
                x2 = int(box.get('xmax', 0) * self.scale_x)
                y1 = int(box.get('ymin', 0) * self.scale_y)
                y2 = int(box.get('ymax', 0) * self.scale_y)
                
                # Высота в индексах Z
                z_center_metric = (box.get('zmin', 0) + box.get('zmax', 0)) / 2.0
                z_idx = z_center_metric * self.scale_z
                
                # Нормализация высоты к [0, 1] относительно высоты домена
                z_norm = z_idx / self.max_z_idx
                
                x1, x2 = np.clip([x1, x2], 0, shape[1])
                y1, y2 = np.clip([y1, y2], 0, shape[0])
                
                # Записываем высоту источника
                h_map[y1:y2, x1:x2] = z_norm
            except Exception as e:
                print(f"Exeption in get_source_height_map: {e}")

        return h_map
    

def read_tecplot_robust(filepath, expected_shape=(DATA_CONFIG['nz'], DATA_CONFIG['ny'], DATA_CONFIG['nx'])):
    header_rows = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() and (line.strip()[0].isdigit() or line.strip()[0] == '-'):
                header_rows = i; break
    try:
        df = pd.read_csv(filepath, sep=r'\s+', skiprows=header_rows, names=['x', 'y', 'z', 'val'], engine='python')
        if len(df) != np.prod(expected_shape): return None
        return df['val'].values.reshape(expected_shape)
    except Exception as e:
        print(f" Exception in read_tecplot_robust {e}")

        
        
        
def process_experiment_25d(build_path, output_path, save_root):
    """
    Обрабатывает один эксперимент, генерируя 2.5D признаки.
    """
    # 0. Инициализация имен
    build_name = os.path.basename(os.path.normpath(build_path))
    output_name = os.path.basename(os.path.normpath(output_path))
    
    config_file = os.path.join(output_path, 'config.txt')
    if not os.path.exists(config_file):
        return

    try:
        cfg = RobustConfigParser(config_file)
    except Exception:
        return

    # 1. Загрузка геометрии (LAD)
    lad_files = glob.glob(os.path.join(output_path, 'common', '3d', 'LAD*.plt'))
    if not lad_files:
        return
    
    # Ожидаемая форма (NZ, NY, NX)
    grid_shape = (DATA_CONFIG['nz'], DATA_CONFIG['ny'], DATA_CONFIG['nx'])
    lad_tensor = read_tecplot_robust(lad_files[0], expected_shape=grid_shape)
    if lad_tensor is None:
        return

    # Маскирование зданий и растительности
    buildings_mask_3d = (lad_tensor < DATA_CONFIG['building_threshold'])
    trees_3d = lad_tensor.copy()
    trees_3d[buildings_mask_3d] = 0.0
    trees_3d[trees_3d < 0] = 0.0
    
    # --- Генерация признаков (Features) ---
    
    # F1: Карта высот зданий (нормированная)
    z_indices = np.arange(DATA_CONFIG['nz']).reshape(-1, 1, 1)
    build_h_map = np.max(buildings_mask_3d * z_indices, axis=0).astype(np.float32)
    build_h_map /= float(DATA_CONFIG['nz'])
    
    # F2: SDF зданий (нормированная по ширине сетки)
    footprint = np.max(buildings_mask_3d, axis=0)
    sdf_2d = distance_transform_edt(~footprint).astype(np.float32)
    sdf_2d /= float(DATA_CONFIG['nx'])
    
    # F3: LAI (интегральная плотность листвы)
    lai_map = np.sum(trees_3d, axis=0).astype(np.float32)

    # --- Обработка Ветра ---
    u = cfg.wind_params['dPdx']
    v = cfg.wind_params['dPdy']
    
    magnitude = np.sqrt(u**2 + v**2)
    if magnitude > DATA_CONFIG['wind_eps']:
        cos_theta = u / magnitude
        sin_theta = v / magnitude
    else:
        cos_theta, sin_theta, magnitude = 0.0, 0.0, 0.0

    # Глобальный вектор (FFN input)
    wind_global = np.array([cos_theta, sin_theta, magnitude], dtype=np.float32)

    # Локальные карты ветра (поля)
    map_shape = (DATA_CONFIG['ny'], DATA_CONFIG['nx'])
    wind_cos_map = np.full(map_shape, cos_theta, dtype=np.float32)
    wind_sin_map = np.full(map_shape, sin_theta, dtype=np.float32)
    wind_mag_map = np.full(map_shape, magnitude, dtype=np.float32)

    # --- Обработка Трейсеров (Концентраций) ---
    conc_files = get_filenames(os.path.join(output_path, 'stat-3d/'), DATA_CONFIG['conc_file_pattern'])
    
    for c_file in conc_files:
        try:
            tracer_id = int(re.search(DATA_CONFIG['tracer_id_regex'], c_file).group(1)) + 1
        except Exception:
            continue
            
        full_c_path = os.path.join(output_path, 'stat-3d/', c_file)
        conc_3d = read_tecplot_robust(full_c_path, expected_shape=grid_shape)
        if conc_3d is None:
            continue
            
        conc_3d[conc_3d < DATA_CONFIG['building_threshold']] = 0.0

        # F4: Высота источника
        source_h_map = cfg.get_source_height_map(tracer_id, shape=map_shape)
        
        # F5: SDF источника
        source_sdf = distance_transform_edt(source_h_map <= 0).astype(np.float32)
        source_sdf /= float(DATA_CONFIG['nx'])

        # Сборка входного тензора X (8 каналов)
        x_tensor = np.stack([
            build_h_map,     # 0
            sdf_2d,          # 1
            lai_map,         # 2
            source_h_map,    # 3
            source_sdf,      # 4
            wind_cos_map,    # 5
            wind_sin_map,    # 6
            wind_mag_map     # 7
        ], axis=0)

        # Сборка выходного тензора Y
        target_slices = []
        for z in DATA_CONFIG['target_z_indices']:
            target_slices.append(conc_3d[z, :, :])
        
        # Добавляем вертикальное среднее
        target_slices.append(np.mean(conc_3d, axis=0))
        y_tensor = np.stack(target_slices, axis=0).astype(np.float32)

        # Сохранение
        save_fname = f"{build_name}__{output_name}__tracer{tracer_id}.npz"
        np.savez_compressed(
            os.path.join(save_root, save_fname), 
            x=x_tensor, 
            y=y_tensor, 
            wind=wind_global
        )

# --- ТОЧКА ВХОДА ---
if __name__ == "__main__":
    os.makedirs(DATA_CONFIG['save_root'], exist_ok=True)
    
    # Поиск папок экспериментов (пример: output_*)
    search_path = os.path.join(DATA_CONFIG['raw_root'], '2026_01_19_500_25d', 'output_*')
    all_outputs = glob.glob(search_path)
    all_outputs.sort(key=os.path.getctime)

    print(f"Processing {len(all_outputs)} experiments into hybrid 2.5D dataset...")

    for out_path in tqdm(all_outputs):
        try:
            process_experiment_25d(
                build_path=os.path.dirname(out_path), 
                output_path=out_path, 
                save_root=DATA_CONFIG['save_root']
            )
        except Exception as e:
            print(f"Error processing {out_path}: {e}")
            
            
            
            
            
# def process_experiment_25d(build_path, output_path):
#     build_name = os.path.basename(os.path.normpath(build_path))
#     output_name = os.path.basename(os.path.normpath(output_path))
    
#     config_file = os.path.join(output_path, 'config.txt')
#     if not os.path.exists(config_file): 
#         print(f"Skipping {output_name}: No config.txt")
#         return

#     try: 
#         cfg = RobustConfigParser(config_file)
#     except Exception as e: 
#         print(f"Config Parse Error in {output_name}: {e}")
#         return

#     # --- 1. Обработка Геометрии (Common) ---
#     lad_files = glob.glob(os.path.join(output_path, 'common', '3d', 'LAD*.plt'))
#     if not lad_files: return
#     lad_tensor = read_tecplot_robust(lad_files[0]) # (32, 128, 128)
#     if lad_tensor is None: return

#     # Разделяем здания и деревья
#     buildings_mask_3d = (lad_tensor < -9000)
#     trees_3d = lad_tensor.copy()
#     trees_3d[buildings_mask_3d] = 0.0
#     trees_3d[trees_3d < 0] = 0.0 # Убираем мусор
    
#     # === FEATURE 1: Building Height Map (2D) ===
#     # Находим максимальный индекс Z, где есть здание. 
#     # argmax вернет 0, если зданий нет. 
#     # Трюк: умножаем маску на индексы Z
#     z_indices = np.arange(GRID_DIMS['z']).reshape(-1, 1, 1)
#     # Массив высот (в индексах)
#     buildings_z = buildings_mask_3d * z_indices
#     # Максимальная высота здания в точке (y, x)
#     build_h_map = np.max(buildings_z, axis=0).astype(np.float32)
#     # Нормализуем к [0, 1]
#     build_h_map /= GRID_DIMS['z']

#     # === FEATURE 2: Building SDF (2D) ===
#     # Берем футпринт зданий (где есть здание хоть на какой-то высоте)
#     footprint_mask = np.max(buildings_mask_3d, axis=0) # (128, 128) Bool
#     air_mask_2d = ~footprint_mask
#     sdf_2d = distance_transform_edt(air_mask_2d).astype(np.float32)
#     sdf_2d /= 128.0 # Нормализация

#     # === FEATURE 3: LAI Map (2D) ===
#     # Интеграл плотности листвы по высоте
#     lai_map = np.sum(trees_3d, axis=0).astype(np.float32)
#     # Можно логарифмировать LAI, так как он может быть большим, но пока оставим linear

#     # --- 2. Обработка Трейсеров ---
#     file_pattern = r'C\[\d+\]-avg-\.plt'
#     conc_files = get_filenames(output_path + '/stat-3d/', file_pattern)
    
#     for c_file in conc_files:
#         try: 
#             tracer_id = int(re.search(r'C\[(\d+)\]', os.path.basename(output_path + '/stat-3d/' + c_file)).group(1)) + 1
#         except: 
#             continue
        
#         conc_3d = read_tecplot_robust(output_path + '/stat-3d/' + c_file)
#         if conc_3d is None: continue
#         conc_3d[conc_3d < -9000] = 0.0

#         # === FEATURE 4: Source Height Map (2D) ===
#         source_h_map = cfg.get_source_height_map(tracer_id, shape=(128, 128))
#         source_mask_inv = (source_h_map <= 0) 
#         # === FEATURE 5: Source SDF (2D) ===
#         source_sdf_2d = distance_transform_edt(source_mask_inv).astype(np.float32)        
#         source_sdf_2d /= 128.0 # Нормализация
        
#         # === FEATURES 6 & 7: Wind ===
#         wind_x = np.full((128, 128), cfg.wind_params['dPdx'], dtype=np.float32)
#         wind_y = np.full((128, 128), cfg.wind_params['dPdy'], dtype=np.float32)

#         # Сборка входного тензора X (Channels=6)
#         x_tensor = np.stack([
#             build_h_map, 
#             sdf_2d, 
#             lai_map, 
#             source_h_map, 
#             source_sdf_2d,
#             wind_x, 
#             wind_y
#         ], axis=0) # (7, 128, 128)

#         # --- 3. Сборка Таргета Y (Multi-Channel 2D) ---
#         target_slices = []
        
#         # Срезы по высоте
#         for z_idx in TARGET_Z_INDICES:
#             # Clip index to be safe
#             safe_z = min(z_idx, conc_3d.shape[0]-1)
#             target_slices.append(conc_3d[safe_z, :, :])
            
#         # Среднее по столбу (Vertical Mean)
#         col_mean = np.mean(conc_3d, axis=0)
#         target_slices.append(col_mean)
        
#         # Стек выходов
#         y_tensor = np.stack(target_slices, axis=0).astype(np.float32) 
#         # y_tensor shape: (5, 128, 128) -> [Z1, Z5, Z8, Z25, Mean]

#         # Сохранение
#         save_fname = f"{build_name}__{output_name}__tracer{tracer_id}.npz"
#         np.savez_compressed(os.path.join(SAVE_ROOT, save_fname), x=x_tensor, y=y_tensor)

# os.makedirs(SAVE_ROOT, exist_ok=True)
# all_outputs = glob.glob(os.path.join(RAW_ROOT, '2026_01_19_500_25d', 'output_*')) # '2026_01_19_500_25d', '2026_01_27_500_25d'
# all_outputs.sort(key=os.path.getctime)

# print(f"Processing {len(all_outputs)} experiments into 2.5D dataset...")

# for out_path in tqdm(all_outputs):
#     try:
#         print(out_path)
#         process_experiment_25d(os.path.dirname(out_path), out_path)
#     except Exception as e:
#         print(f"Error {out_path}: {e}")