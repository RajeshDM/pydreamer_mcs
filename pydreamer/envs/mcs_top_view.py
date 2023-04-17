from gym import spaces
import numpy as np
import cv2
import gym
from gym import spaces
import machine_common_sense as mcs
import random
from pydreamer.envs.mcs_constants import *
import os

class MCSTopView(gym.Env):
    def __init__(self) -> None:
        super(MCSTopView, self).__init__()

        self.controller = mcs.create_controller(unity_app_file_path=UNITY_PATH,
                                                config_file_or_dict='./mcs_config.ini')

        assert self.controller != None

        dataset_dir = 'scenes/moving_target_10k'

        self.all_scenes = [
            os.path.join(dataset_dir, one_scene)
            for one_scene in sorted(os.listdir(dataset_dir))
            if one_scene.endswith(".json")
        ]

        self.scene_name = random.choice(self.all_scenes)
        self.scene_data = mcs.load_scene_json_file(self.scene_name)

        # Skipping initial 360 rotation
        self.scene_data['goal']['action_list'] = []
        for obj in self.scene_data['objects']:
            if obj['type'] == 'soccer_ball':
                obj['forces'][0]['stepBegin'] = 0
                obj['forces'][0]['stepEnd'] = 0

        self.output = self.controller.start_scene(self.scene_data)

        self.new_dimension = 84
        self.action_space = spaces.Discrete(7)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.new_dimension, self.new_dimension, 2), dtype=np.float32)
        self.action_list = {0:"MoveAhead", 1:"MoveLeft", 2:"MoveRight", 3:"MoveBack", \
                            4:"PickupObject,objectId=", 5:"RotateLeft", 6:"RotateRight"}

    def step(self, action):
        done = False
        solved = False
        stepped_on_lava = False
        reward = -0.5

        ms_action = self.action_list[action.item()]
        if action.item() == 4:
            ms_action += self.soccer_id
        self.output = self.controller.step(ms_action)

        if action.item() == 4 and self.output.return_status == "SUCCESSFUL":
            reward += 250
            done = True
            solved = True

        if self.output.steps_on_lava == 0:
            for obj in self.output.object_list:
                if obj.uuid == self.soccer_id:
                    if not obj.visible:
                        reward -= 2

                    if self.prev_distance - obj.distance_in_steps > 0:
                        reward += 0.5
                    else:
                        reward -= 0.5
                    self.prev_distance = obj.distance_in_steps
                    # self.curr_distance = obj.distance_in_steps
                    # if self.prev_distance < self.curr_distance:
                    #     reward -= 2
                    # self.prev_distance = self.curr_distance

            if self.output.step_number > 128:
                done = True

            if not done:
                self.process_observation()
        else:
            reward -= 75
            done = True
            stepped_on_lava = True

        return self.observation, reward, done, {"solved":solved, "scene_name": self.scene_name, "stepped_on_lava": stepped_on_lava}

    def reset(self):
        self.end()

        self.scene_name = random.choice(self.all_scenes)
        self.scene_data = mcs.load_scene_json_file(self.scene_name)

        self.scene_data['goal']['action_list'] = []
        self.soccer_id = self.scene_data['goal']['metadata']['target']['id']
        for obj in self.scene_data['objects']:
            if obj['type'] == 'soccer_ball':
                obj['forces'][0]['stepBegin'] = 0
                obj['forces'][0]['stepEnd'] = 0
        self.output = self.controller.start_scene(self.scene_data)

        for obj in self.output.object_list:
            if obj.uuid == self.soccer_id:
                self.soccer_rgb = list(obj.segment_color.values())
                self.prev_distance = obj.distance_in_steps

        self.process_observation()
        return self.observation

    def end(self):
        self.controller.end_scene()

    def close(self):
        self.controller.stop_simulation()

    def process_observation(self):
        self.rgb_im = np.array(self.output.image_list[-1])
        self.rgb_im = cv2.resize(self.rgb_im, (self.new_dimension, self.new_dimension))
        self.gray_img = cv2.cvtColor(self.rgb_im, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        self.depth_map = cv2.resize(self.output.depth_map_list[-1], (self.new_dimension, self.new_dimension)).astype(np.float32) / 150.0
        # self.mask = cv2.resize(np.array(self.output.object_mask_list[-1]), (self.new_dimension, self.new_dimension))
        # self.soccer_mask = np.all(self.mask == self.soccer_rgb, axis=-1).astype(np.float32)

        self.observation = np.stack((self.gray_img, \
                                     self.depth_map), axis=-1)

    def render(self, mode):
        pass


if __name__ == "__main__":

    from stable_baselines3.common.env_checker import check_env

    env = MCSMTarget()
    check_env(env)

    from gym.envs.registration import register

    register(
        id="MCS-v5",
        entry_point=MCSMTarget,
        max_episode_steps=128,
    )

    print("starting tests")
    for i in range(5):
        env.reset()
        print(f"evaluation # {i}")
        done = False
        acc_reward  = 0
        while not done:
            obs, reward, done, info = env.step(0)
            acc_reward += reward
        print(acc_reward)
