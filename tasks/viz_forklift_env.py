# test version

from isaacgym import gymutil, gymtorch, gymapi
from isaacgymenvs.tasks.base.vec_task import VecTask

import numpy as np
import os
import torch


class ForkliftEnv(VecTask):
    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless):
        # config를 self.cfg에 저장
        self.cfg = config
        self.env_spacing = 2.0  # 환경 간의 간격을 명시적으로 초기화
        self.envs_per_row = int(np.sqrt(config["env"]["numEnvs"]))  # 환경 배치 계산
        self.max_episode_length = config["env"].get("max_episode_length", 1000)  # 기본값 1000
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)

    def create_sim(self):
        if hasattr(self, 'sim') and self.sim is not None:
            print("Sim already exists.")
            return

        # 기존 초기화 코드
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

        self.create_envs()


    def create_envs(self):
        """환경 내에서 에셋 배치"""
        lower = gymapi.Vec3(-self.env_spacing / 2, -self.env_spacing / 2, 0.0)
        upper = gymapi.Vec3(self.env_spacing / 2, self.env_spacing / 2, self.env_spacing)

        self.envs = []
        self.handles = []
        for i in range(self.cfg["env"]["numEnvs"]):
            env = self.gym.create_env(self.sim, lower, upper, self.envs_per_row)
            self.envs.append(env)

            # 에셋 배치
            pose = gymapi.Transform()
            pose.p.z = 2.0  # 높이 설정
            handle = self.gym.create_actor(env, self.pallet_asset, pose, "pallet", i, 1)
            self.handles.append(handle)

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
            "numEnvs": 1,
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

    # 렌더링 루프
    while not env.gym.query_viewer_has_closed(env.viewer):
        env.render()
        actions = torch.zeros((config["env"]["numEnvs"], config["env"]["numActions"]), device="cpu")
        env.step(actions)


if __name__ == "__main__":
    run_forklift_env()
