import os
import numpy as np
from ._base_dataset import _BaseDataset
from .. import utils
from .. import _timing
from ..utils import TrackEvalException

# 3D BEV IoU 계산을 위한 shapely 임포트
try:
    from shapely.geometry import Polygon
except ImportError:
    print('Shapely library not found. Please install shapely for 3D BEV IoU calculation.')
    raise

class AICityChallengeSingleFile(_BaseDataset):
    """
    Dataset class for AI City Challenge, simplified for a single GT and prediction file.
    Assumes two files: ground_truth.txt and predict.txt.
    """

    @staticmethod
    def get_default_dataset_config():
        """Default class config values"""
        code_path = utils.get_code_path()
        default_config = {
            # 파일 경로를 직접 지정
            'GT_FILE_PATH': None,      # 예: '/path/to/ground_truth.txt'
            'TRACKER_FILE_PATH': None, # 예: '/path/to/predict.txt'
            'CLASSES_TO_EVAL': ['Person', 'Forklift', 'NovaCarter', 'Transporter', 'FourierGR1T2', 'AgilityDigit'],
            'PRINT_CONFIG': True,
        }
        return default_config

    def __init__(self, config=None):
        """Initialise dataset, checking that all required files are present"""
        super().__init__()
        # 설정 초기화
        self.config = utils.init_config(config, self.get_default_dataset_config(), self.get_name())

        # 평가할 클래스 목록 및 ID 매핑 정의
        self.class_list = [cls.lower() for cls in self.config['CLASSES_TO_EVAL']]
        self.class_name_to_class_id = {
            'person': 0, 'forklift': 1, 'novacarter': 2, 'transporter': 3,
            'fouriergr1t2': 4, 'agilitydigit': 5
        }
        
        # 파일 경로 확인
        self.gt_file = self.config['GT_FILE_PATH']
        self.tracker_file = self.config['TRACKER_FILE_PATH']
        if self.gt_file is None or not os.path.isfile(self.gt_file):
            raise TrackEvalException(f"GT file not found at: {self.gt_file}")
        if self.tracker_file is None or not os.path.isfile(self.tracker_file):
            raise TrackEvalException(f"Tracker file not found at: {self.tracker_file}")

        # GT 파일로부터 시퀀스 정보(scene_id, 프레임 수) 추출
        self.seq_list, self.seq_lengths = self._get_seq_info()
        if not self.seq_list:
            raise TrackEvalException('No sequences found in GT file.')
            
        # 데이터 미리 로드
        self.gt_data = self._load_data(self.gt_file, is_gt=True)
        # TrackEval 프레임워크는 여러 트래커 비교를 가정하므로, 단일 예측 파일을 'prediction'이라는 이름의 트래커로 간주
        self.tracker_list = ['prediction']
        self.tracker_to_disp = {'prediction': 'prediction'}
        self.tracker_data = {
            'prediction': self._load_data(self.tracker_file, is_gt=False)
        }
        self.output_fol = os.path.join(os.getcwd(), 'trackeval_output')  # 또는 원하는 경로
        self.output_sub_fol = 'default_subfolder'  # 필요에 따라 변경

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _get_seq_info(self):
        """GT 파일 전체를 읽어 모든 scene(시퀀스)과 각 scene의 길이를 파악합니다."""
        seq_list = set()
        seq_lengths = {}
        with open(self.gt_file) as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if not parts: continue
                    scene_id = parts[0]
                    frame_id = int(parts[3])
                    seq_list.add(scene_id)
                    seq_lengths[scene_id] = max(seq_lengths.get(scene_id, -1), frame_id)
                except (IndexError, ValueError) as e:
                    raise TrackEvalException(f"Error parsing GT file line: {line.strip()}. Reason: {e}")
        return sorted(list(seq_list)), seq_lengths
    
    def _load_data(self, file_path, is_gt):
        """
        하나의 큰 txt 파일을 읽어 scene_id를 키로 하는 딕셔너리로 데이터를 정리합니다.
        """
        data_by_scene = {}
        with open(file_path) as f:
            for line in f:
                try:
                    parts = line.strip().split()
                    if not parts: continue
                    scene_id = parts[0]
                    frame_id = int(parts[3])
                    
                    if scene_id not in data_by_scene:
                        data_by_scene[scene_id] = {}
                    
                    if frame_id not in data_by_scene[scene_id]:
                        data_by_scene[scene_id][frame_id] = []
                    
                    float_parts = [float(p) for p in parts[1:]]
                    data_by_scene[scene_id][frame_id].append(float_parts)
                except (IndexError, ValueError) as e:
                    raise TrackEvalException(f"Error parsing data file line: {line.strip()}. Reason: {e}")

        return data_by_scene

    def _load_raw_file(self, tracker, seq, is_gt):
        """
        미리 로드된 데이터에서 특정 시퀀스(scene)에 대한 원본 데이터를 TrackEval 형식으로 변환합니다.
        """
        if is_gt:
            seq_data = self.gt_data.get(seq, {})
        else:
            seq_data = self.tracker_data.get(tracker, {}).get(seq, {})

        num_timesteps = self.seq_lengths[seq]
        data_keys = ['ids', 'classes', 'dets']
        if not is_gt:
            data_keys.append('tracker_confidences')
        raw_data = {key: [[] for _ in range(num_timesteps)] for key in data_keys}
        # ndarray로 변환하기 위해 빈 리스트 대신 empty array로 초기화
        raw_data['dets'] = [np.empty((0, 7)) for _ in range(num_timesteps)]
        raw_data['ids'] = [np.empty(0).astype(int) for _ in range(num_timesteps)]
        raw_data['classes'] = [np.empty(0).astype(int) for _ in range(num_timesteps)]
        if not is_gt:
            raw_data['tracker_confidences'] = [np.empty(0) for _ in range(num_timesteps)]


        for t in range(num_timesteps):
            time_key = t # 프레임 ID는 0-based
            if time_key in seq_data:
                time_data = np.asarray(seq_data[time_key], dtype=np.float64)
                
                raw_data['classes'][t] = np.atleast_1d(time_data[:, 0]).astype(int)
                raw_data['ids'][t] = np.atleast_1d(time_data[:, 1]).astype(int)
                raw_data['dets'][t] = np.atleast_2d(time_data[:, 3:10])
                
                if not is_gt:
                    if time_data.shape[1] > 10:
                         raw_data['tracker_confidences'][t] = np.atleast_1d(time_data[:, 10])
                    else:
                        raw_data['tracker_confidences'][t] = np.ones(len(time_data))

        if is_gt:
            key_map = {'ids': 'gt_ids', 'classes': 'gt_classes', 'dets': 'gt_dets'}
        else:
            key_map = {'ids': 'tracker_ids', 'classes': 'tracker_classes', 'dets': 'tracker_dets'}
        
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        raw_data['num_timesteps'] = num_timesteps
        raw_data['seq'] = seq
        return raw_data

    # get_preprocessed_seq_data 와 _calculate_similarities 메서드는 이전과 동일하게 유지됩니다.
    # (코드 중복을 피하기 위해 여기에 다시 붙여넣지는 않았지만, 실제 파일에는 포함되어야 합니다.)
    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """
        특정 클래스에 대해 데이터를 전처리합니다.
        distractor 처리 등 복잡한 로직 없이, 해당 클래스의 데이터만 필터링합니다.
        """
        cls_id = self.class_name_to_class_id[cls.lower()]
        
        self._check_unique_ids(raw_data)

        data_keys = ['gt_ids', 'tracker_ids', 'gt_dets', 'tracker_dets', 'tracker_confidences', 'similarity_scores']
        data = {key: [None] * raw_data['num_timesteps'] for key in data_keys}
        
        unique_gt_ids = set()
        unique_tracker_ids = set()
        num_gt_dets = 0
        num_tracker_dets = 0

        for t in range(raw_data['num_timesteps']):
            # GT 필터링
            gt_ids = raw_data['gt_ids'][t]
            gt_dets = raw_data['gt_dets'][t]
            gt_classes = raw_data['gt_classes'][t]
            gt_to_keep_mask = (gt_classes == cls_id)
            
            data['gt_ids'][t] = gt_ids[gt_to_keep_mask]
            data['gt_dets'][t] = gt_dets[gt_to_keep_mask, :]
            
            # Tracker 필터링
            tracker_ids = raw_data['tracker_ids'][t]
            tracker_dets = raw_data['tracker_dets'][t]
            tracker_classes = raw_data['tracker_classes'][t]
            tracker_confidences = raw_data['tracker_confidences'][t]
            tracker_to_keep_mask = (tracker_classes == cls_id)
            
            data['tracker_ids'][t] = tracker_ids[tracker_to_keep_mask]
            data['tracker_dets'][t] = tracker_dets[tracker_to_keep_mask, :]
            data['tracker_confidences'][t] = tracker_confidences[tracker_to_keep_mask]
            
            data['similarity_scores'][t] = self._calculate_similarities(data['gt_dets'][t], data['tracker_dets'][t])

            unique_gt_ids.update(data['gt_ids'][t])
            unique_tracker_ids.update(data['tracker_ids'][t])
            num_gt_dets += len(data['gt_ids'][t])
            num_tracker_dets += len(data['tracker_ids'][t])

        # ID 재라벨링
        if unique_gt_ids:
            gt_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(list(unique_gt_ids)))}
            for t in range(raw_data['num_timesteps']):
                if len(data['gt_ids'][t]) > 0:
                    data['gt_ids'][t] = np.array([gt_id_map[old_id] for old_id in data['gt_ids'][t]], dtype=int)
        
        if unique_tracker_ids:
            tracker_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(list(unique_tracker_ids)))}
            for t in range(raw_data['num_timesteps']):
                if len(data['tracker_ids'][t]) > 0:
                    data['tracker_ids'][t] = np.array([tracker_id_map[old_id] for old_id in data['tracker_ids'][t]], dtype=int)
        
        data['num_tracker_dets'] = num_tracker_dets
        data['num_gt_dets'] = num_gt_dets
        data['num_tracker_ids'] = len(unique_tracker_ids)
        data['num_gt_ids'] = len(unique_gt_ids)
        data['num_timesteps'] = raw_data['num_timesteps']
        data['seq'] = raw_data['seq']

        self._check_unique_ids(data, after_preproc=True)
        return data

    
    def _calculate_similarities(self, gt_dets_t, tracker_dets_t):
        """3D 박스의 BEV(Bird's-Eye-View) IoU를 계산합니다."""
        if gt_dets_t.shape[0] == 0 or tracker_dets_t.shape[0] == 0:
            return np.empty((gt_dets_t.shape[0], tracker_dets_t.shape[0]))
            
        gt_polys = self._get_bev_polygons(gt_dets_t)
        tracker_polys = self._get_bev_polygons(tracker_dets_t)

        similarity_scores = np.zeros((len(gt_polys), len(tracker_polys)))
        for i, gt_poly in enumerate(gt_polys):
            for j, trk_poly in enumerate(tracker_polys):
                intersection_area = gt_poly.intersection(trk_poly).area
                union_area = gt_poly.union(trk_poly).area
                if union_area > 0:
                    similarity_scores[i, j] = intersection_area / union_area
        
        return similarity_scores

    @staticmethod
    def _get_bev_polygons(dets):
        """[x, y, z, w, l, h, yaw] 형식의 det에서 BEV 폴리곤을 생성합니다."""
        polygons = []
        for det in dets:
            x, y, _, w, l, _, yaw = det

            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
            corners = np.array([[-w/2, -l/2], [-w/2, l/2], [w/2, l/2], [w/2, -l/2]])
            transformed_corners = corners @ R.T + np.array([x, y])
            polygons.append(Polygon(transformed_corners))
        return polygons