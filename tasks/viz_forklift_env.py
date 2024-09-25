from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask

import numpy as np
import os
import torch


class ForkliftEnv(VecTask):
    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless):
        self.cfg = config
        self.env_spacing = 50.0  # 환경 간의 간격 설정
        self.num_pallets = 9  # 팔레트의 개수이자 환경의 수 설정
        self.max_episode_length = config["env"].get("max_episode_length", 1000)  # 기본값 1000
        self.device = rl_device
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)

    def create_sim(self):
        if hasattr(self, 'sim') and self.sim is not None:
            print("Sim already exists.")
            return

        # 초기화 코드
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.dt = 1 / 60.0
        sim_params.substeps = 2

        graphics_device_id = 0
        headless = False

        self.sim = self.gym.create_sim(0, graphics_device_id, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None:
            print("Error: 시뮬레이션을 생성할 수 없습니다.")
            return

        # 바닥 및 에셋 생성
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        asset_file = "forklift/Pallet_A1/Pallet_A1.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        self.pallet_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        if self.pallet_asset is None:
            print("Error: 에셋을 불러올 수 없습니다.")
            return

        # 환경 생성 및 팔레트 배치
        self._create_envs_with_pallets(self.num_pallets, self.env_spacing)

    def _create_envs_with_pallets(self, num_envs, spacing):
        """각 환경을 3x3 배열로 배치하고, 각 환경의 중심에 팔레트를 놓음"""
        num_per_row = 3  # 한 줄에 3개의 환경을 배치 (3x3 배열)
        
        self.envs = []
        self.handles = []
    
        # 환경의 좌표를 3x3 배열로 배치하기 위한 기준점 계산
        for i in range(num_envs):  
            # 3x3 그리드에서 x, y 좌표 계산
            row = i // num_per_row  # 현재 행 번호
            col = i % num_per_row   # 현재 열 번호
    
            # 환경의 좌표 오프셋 계산
            env_offset_x = col * spacing  # 열에 따라 x축으로 이동
            env_offset_y = row * spacing  # 행에 따라 y축으로 이동
    
            # 환경 생성 (여기서 오프셋을 사용하여 환경 위치 지정)
            env_handle = self.gym.create_env(self.sim, gymapi.Vec3(env_offset_x - spacing/2, env_offset_y - spacing/2, 0.0),
                                             gymapi.Vec3(env_offset_x + spacing/2, env_offset_y + spacing/2, spacing), num_envs)
            self.envs.append(env_handle)
    
            # 팔레트의 위치 설정 (각 환경의 중심에 위치)
            start_pose = gymapi.Transform()
            start_pose.p.x = 0.0  # 팔레트는 각 환경의 중심에 위치
            start_pose.p.y = 0.0
            start_pose.p.z = 0.5  # 팔레트의 z 위치
    
            # 액터 생성 (팔레트)
            handle = self.gym.create_actor(env_handle, self.pallet_asset, start_pose, "pallet", i, 1)
    
            # 팔레트 크기 조정 (작게 설정)
            scale = 0.05  # 팔레트 크기를 더 작게 설정
            self.gym.set_actor_scale(env_handle, handle, scale)
            self.handles.append(handle)



    def set_camera(self):
        """카메라 위치를 설정하여 더 넓은 환경을 볼 수 있게 함"""
        cam_pos = gymapi.Vec3(500, 500, 200)  # 카메라를 환경에서 멀리 떨어진 위치로 설정
        cam_target = gymapi.Vec3(0, 0, 0)  # 카메라가 환경 중심을 바라보도록 설정
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def pre_physics_step(self, actions):
        """물리 엔진 업데이트 전 동작을 정의"""
        pass  # 필요시 실제 동작 추가

    def post_physics_step(self):
        """물리 엔진 업데이트 후 동작을 정의"""
        pass  # 필요시 실제 동작 추가

    def step(self, actions):
        """환경의 한 스텝을 진행"""
        super().step(actions)
        # 이 부분에서 추가적인 스텝 로직을 구현할 수 있습니다.

    def reset(self):
        """환경 리셋"""
        return super().reset()

    def render(self):
        """환경을 렌더링"""
        super().render()


# 기본 설정과 함께 환경 실행
def run_forklift_env():
    config = {
        "env": {
            "numEnvs": 9,  # 9개의 환경을 설정, 각 환경에 팔레트 하나씩 배치
            "numActions": 4,
            "numObservations": 10
        },
        "sim": {
            "use_gpu_pipeline": False,
            "up_axis": "z",
            "dt": 1 / 60.0,  # 시뮬레이션 시간 스텝 크기를 정의
            "gravity": [0.0, 0.0, -9.81]  # 중력 설정 (Z축 방향으로 중력)
        },
        "physics_engine": "physx"
    }

    env = ForkliftEnv(config, rl_device="cpu", sim_device="cpu", graphics_device_id=0, headless=False)
    env.create_sim()

    # 카메라 위치 설정
    env.set_camera()

    # 렌더링 루프
    while not env.gym.query_viewer_has_closed(env.viewer):
        env.render()
        actions = torch.zeros((config["env"]["numEnvs"], config["env"]["numActions"]), device="cpu")
        env.step(actions)


if __name__ == "__main__":
    run_forklift_env()
