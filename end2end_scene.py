import argparse
import configparser
import json
import os
import pickle
import sys
from pathlib import Path

#import machine_common_sense as mcs
import yaml

import torch  # isort:skip

print(torch.__version__)

from pathlib import Path

#import opics_inter

#os.environ["OPICS_HOME"] = str(
#    Path(os.path.dirname(opics_inter.__file__)).parent.absolute()
#)
#from opics_inter.common.utils.utils import get_scene_type


def get_unity_path(cfg_dir):
    yaml_path = os.path.join(cfg_dir, "unity_path.yaml")
    if not os.path.exists(yaml_path):
        raise FileNotFoundError("unity_path.yaml missing from cfg dir")
    with open(yaml_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return os.path.join(config["unity_path"])


def get_config_ini_path(cfg_dir):
    print(f"cwd: {os.getcwd()}")
    print("cfg_dir: ", cfg_dir)
    ini_path = os.path.join(cfg_dir, "default.ini")
    print("ini_path: ", ini_path)
    if not os.path.exists(ini_path):
        raise FileNotFoundError("default.ini missing from cfg dir")
    return ini_path


def get_level_from_config_ini(config_ini_path):
    config_ini = configparser.ConfigParser()
    config_ini.read(config_ini_path)
    return config_ini["MCS"]["metadata"]


# scene status
IN_PROGRESS = "IN_PROGRESS"
NOT_ATTEMPTED = "ready"
IN_PROGRESS_SCENE_STARTED = IN_PROGRESS + "_SCENE_STARTED"
IN_PROGRESS_CONTROLLER_LAUNCHED = IN_PROGRESS + "_CONTROLLER_LAUNCHED"
IN_PROGRESS_CONTROLLER_UP = IN_PROGRESS + "_CONTROLLER_UP"
IN_PROGRESS_SCENE_RUNNING = IN_PROGRESS + "_SCENE_RUNNING"
IN_PROGRESS_SCENE_ASSIGNED = IN_PROGRESS + "_SCENE_ASSIGNED"
COMPLETED = "COMPLETED"

# directives
RETRY_OTHER_TRUN_SESSION = "RETRY_OTHER_TRUN_SESSION"
RETRY_THIS_TRUN_SESSION = "RETRY_THIS_TRUN_SESSION"
RETRY_AFTER_PAUSE = "RETRY_AFTER_PAUSE"
SCENE_FATAL = "SCENE_FATAL"
SESSION_FATAL = "SESSION_FATAL"

CONTROLLER_FAILED_TO_LAUNCH = (
    "CONTROLLER_FAILED_TO_LAUNCH__" + RETRY_OTHER_TRUN_SESSION
)
INITIALIZATION_ERROR = "INITIALIZATION_ERROR__" + RETRY_OTHER_TRUN_SESSION
RUNTIME_ERROR = "RUNTIME_ERROR__" + RETRY_OTHER_TRUN_SESSION

FAILED_TIMEOUT = "FAILED_TIMEOUT__" + RETRY_THIS_TRUN_SESSION

FAILED_EXCEPTION = "EXCEPTION__" + SCENE_FATAL
UNKNOWN_SCENE_TYPE = "UNKNOWN_SCENE_TYPE__" + SCENE_FATAL

FAILED_GPU_MEM = "FAILED_GPU_MEM"
FAILED_GPU_MEM_FATAL = FAILED_GPU_MEM + "__" + SESSION_FATAL
FAILED_GPU_MEM_RETRY = FAILED_GPU_MEM + "__" + RETRY_AFTER_PAUSE
ENVIRONMENT_BAD = "ENVIRONMENT_BAD__" + SESSION_FATAL


class OpicsRunState:
    def __init__(self, scene_path):
        self.scene_path = scene_path
        self.is_systest = False
        self.test_register = None
        self.state = NOT_ATTEMPTED
        self.retry_pause_time = 120

    def set_test_register(self, test_register):
        self.test_register = test_register
        self.is_systest = True

    def is_controller_timed_out(self):
        if self.state == FAILED_TIMEOUT:
            return True
        return False

    def is_session_pointless(self):
        if SESSION_FATAL in self.state:
            return True
        return False

    def should_retry_after_pause(self):
        if RETRY_AFTER_PAUSE in self.state:
            return True
        return False

    def needs_run_attempt(self):
        if self.state == NOT_ATTEMPTED:
            return True
        if RETRY_THIS_TRUN_SESSION in self.state:
            return True
        return False

    def is_optics_run(self):
        return self.is_systest

    def starting_scene(self):
        self.state = IN_PROGRESS_SCENE_STARTED
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def starting_controller(self):
        self.state = IN_PROGRESS_CONTROLLER_LAUNCHED
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def controller_up(self):
        self.state = IN_PROGRESS_CONTROLLER_UP
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def scene_running(self):
        self.state = IN_PROGRESS_SCENE_RUNNING
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def scene_completed(self):
        self.state = COMPLETED
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def bad_environment(self):
        self.state = ENVIRONMENT_BAD
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    # EXCEPTIONS AND ERRORS:
    def initialization_error(self):
        self.state = INITIALIZATION_ERROR
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def error(self):
        self.state = FAILED_EXCEPTION
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def runtime_error(self):
        self.state = RUNTIME_ERROR
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def cuda_memory_error(self):
        self.state = FAILED_GPU_MEM_RETRY
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def controller_timed_out(self):
        self.state = FAILED_TIMEOUT
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def controller_failed_to_launch(self):
        self.state = CONTROLLER_FAILED_TO_LAUNCH
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def convert_exception_to_run_state(self, err, context):
        err_string = str(err)
        if "CUDA" in err_string and "out of memory" in err_string:
            self.cuda_memory_error()
        elif "Time out" in err_string:
            self.controller_timed_out()
        elif self.state == IN_PROGRESS_CONTROLLER_LAUNCHED:
            self.controller_failed_to_launch()
        elif err is RuntimeError:
            self.runtime_error()
        else:
            self.error()

    # SCENE RELATED

    def unknown_scene_type(self):
        self.state = FAILED_EXCEPTION
        if self.is_systest:
            self.test_register.note_scene_state(self.scene_path, self.state)

    def should_tman_assign_scene_in_state(self, run_state):
        if run_state == NOT_ATTEMPTED:
            return True
        elif IN_PROGRESS in run_state:
            return False
        elif run_state == COMPLETED:
            return False
        elif RETRY_AFTER_PAUSE in run_state:
            return True
        elif RETRY_OTHER_TRUN_SESSION in run_state:
            return True
        elif (
            RETRY_THIS_TRUN_SESSION in run_state
        ):  # that session wil be trying again on its own
            return False
        elif (
            SCENE_FATAL in run_state
        ):  #  no reason to think the scene will work on second try
            return False
        elif (
            SESSION_FATAL in run_state
        ):  # the trun session had to abort for some reason, try this scene elsewhere
            return True
        else:
            print(f"unknown state: {run_state}")
            return False

    def show_runs_summary(self, scene_state_histories):
        end_state_counts = {}
        end_state_counts[NOT_ATTEMPTED] = 0
        end_state_counts[IN_PROGRESS_SCENE_ASSIGNED] = 0
        end_state_counts[IN_PROGRESS_SCENE_RUNNING] = 0
        end_state_counts[IN_PROGRESS_SCENE_STARTED] = 0
        end_state_counts[IN_PROGRESS_CONTROLLER_LAUNCHED] = 0
        end_state_counts[IN_PROGRESS_CONTROLLER_UP] = 0
        end_state_counts[COMPLETED] = 0
        end_state_counts[CONTROLLER_FAILED_TO_LAUNCH] = 0
        end_state_counts[INITIALIZATION_ERROR] = 0
        end_state_counts[RUNTIME_ERROR] = 0
        end_state_counts[FAILED_TIMEOUT] = 0
        end_state_counts[FAILED_EXCEPTION] = 0
        end_state_counts[UNKNOWN_SCENE_TYPE] = 0
        end_state_counts[FAILED_GPU_MEM_FATAL] = 0
        end_state_counts[FAILED_GPU_MEM_RETRY] = 0
        end_state_counts[ENVIRONMENT_BAD] = 0
        for ssh in scene_state_histories:
            end_state_counts[ssh.end_state] += 1

        for end_state in end_state_counts:
            end_state_sans_directive = end_state.split("__")[0]
            print(
                f"{end_state_sans_directive.rjust(35)}: {end_state_counts[end_state]}"
            )

    def get_scene_types_from_completed_scenes(self, scene_state_histories):
        scene_types = set()
        for ssh in scene_state_histories:
            if ssh.is_completed():
                scene_types.add(ssh.scene_type)
        return scene_types

    def get_completed_state_histories_for_scene_type(
        self, scene_state_histories, scene_type
    ):
        completed_state_histories = []
        for ssh in scene_state_histories:
            if ssh.is_completed() and ssh.scene_type == scene_type:
                completed_state_histories.append(ssh)
        return completed_state_histories

    def show_scene_timings(self, scene_state_histories):
        scene_types = self.get_scene_types_from_completed_scenes(
            scene_state_histories
        )
        for type in scene_types:
            sshs = self.get_completed_state_histories_for_scene_type(
                scene_state_histories, type
            )
            print(f"{len(sshs)} {type} scene timings:")
            self.show_average_successful_run_time(sshs)
            # self.show_average_total_time(sshs)

    def show_average_successful_run_time(self, scene_state_histories):
        total_time = 0
        complete_count = 0
        for ssh in scene_state_histories:
            if ssh.is_completed():
                complete_count += 1
                total_time += ssh.completion_duration
        average = int(total_time / complete_count)
        mins = int(average / 60)
        sec = int(average % 60)
        print(f"    average succesful run time: {mins} mins {sec} sec")

    def show_average_total_time(self, scene_state_histories):
        total_time = 0
        complete_count = 0
        for ssh in scene_state_histories:
            if ssh.is_completed():
                complete_count += 1
                total_time += ssh.total_duration
        print(
            f"    average run time including retries: {int(total_time / complete_count)}"
        )

    def show_gpu_mem_fail_retry_count(self, scene_state_histories):
        retry_count = 0
        for ssh in scene_state_histories:
            retry_count += ssh.get_gpu_mem_fail_count()
        print(f"\ngpu mem fatal pause count: {retry_count}")


import datetime

from rich import pretty, traceback

traceback.install()
pretty.install()

#print("importing physics_voe_agent")
#import opics_inter
#from opics_inter.inter.inter_agent import InterAgent

#PROJECT_PATH = str(
#    Path(os.path.dirname(opics_inter.__file__)).parent.absolute()
#)


def get_scene_type_from_scene_file(path):
    f = open(path, "r")
    scene_json = json.load(f)
    f.close()
    return scene_json["goal"]["category"]

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_dir", default=f"{PROJECT_PATH}/configs/mcs")
    parser.add_argument("--scene", default="../opics/tests/ptest/eval6_val/interactive_collisions_0001_02.json")
    parser.add_argument("--controller", default="mcs")
    parser.add_argument("--log_dir", default="default")
    parser.add_argument("--scene_type", default=None)
    parser.add_argument("--scene_classifier_only", default=False)
    return parser


def usage():
    print(
        "python run_opics_scene.py --scene <path of json scene file> --log_dir <non_default_log_dir"
    )


if __name__ == "__main__":
    if not "OPICS_HOME" in os.environ:
        print("")
        print(
            "      ERROR - OPICS_HOME not defined.  Please 'export OPICS_HOME=<parent_of_opics_dir>'"
        )
        print("")
        sys.exit()

    args = make_parser().parse_args()
    cfg_dir = args.cfg_dir

    log_dir = args.log_dir
    scene_path = args.scene
    scene_type = args.scene_type
    scene_classifier_only = args.scene_classifier_only

    scene_name = scene_path.split("/")[-1].split(".")[0]
    if "undefined" == scene_path:
        usage()
        sys.exit()

    config_ini_path = get_config_ini_path(cfg_dir)

    print(f"...using config_init_path {config_ini_path}")

    try:
        scene_json = mcs.load_scene_json_file(scene_path)
    except FileNotFoundError:
        print(f"Error : scene file {scene_path} not found")
        sys.exit()
    except Exception as err:
        print(f"problem with scene file {scene_path} : {err} ")
        sys.exit()

    if scene_json == {}:
        print("Scene Config is Empty", scene_path)
        sys.exit()

    level = "level2"
    print("==================================================================")
    print("")
    print(f"METADATA LEVEL: {level}")
    print("")
    print("==================================================================")
    assert level in ["oracle", "level2"]

    print(
        "=========================================================================="
    )
    print(f"       running scene {scene_path}")
    print(
        "=========================================================================="
    )

    # TODO(Mazen): I have commented this out, but idk why we need it
    # can @Jed give feedback on this?
    # scene_type = get_scene_type_from_scene_file(scene_path)

    start_time = datetime.datetime.now()
    run_state = OpicsRunState(scene_path)
    run_state.starting_scene()
    run_state.starting_controller()
    print("------ creating mcs controller ------")
    controller = mcs.create_controller(config_file_or_dict=config_ini_path)
    run_state.controller_up()

    if controller == None:
        raise Exception("controller was initialized incorrectly")

    agent = InterAgent(controller, level,scene_name)

    if scene_type == None :
        scene_type = get_scene_type(scene_name.split("/")[-1])
        print ("Scene type from filename : " , scene_type)

    if scene_classifier_only:
        # store history
        HISTORY_FILE = f"{PROJECT_PATH}/scene_classification_history.pkl"
        history = {}
        try:
            with open(HISTORY_FILE, 'rb') as handle:
                history = pickle.load(handle)
        except:
            pass

        # for scene classifier test runs, we are going to IGNORE scene type
        # because we want to test the classifier can accurately detect the scene type
        classified_scene_type = agent.try_run_scene(
            scene_json, scene_path, run_state, scene_name, classify_scene=True
        )
        if scene_type == classified_scene_type:
            print("Scene classified correctly!")
            history[scene_path] = 'PASS'
        else:
            print(f"Scene mis-classification. Expected {scene_type}, got {classified_scene_type}")
            history[scene_path] = f'EXP={scene_type},ACT={classified_scene_type}'

        with open(HISTORY_FILE, 'wb') as handle:
            pickle.dump(history, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        agent.try_run_scene(
            scene_json, scene_path, run_state, scene_name, scene_type=scene_type
        )

    end_time = datetime.datetime.now()
    total_time = end_time - start_time
    print(f"...total time for scene {total_time}")
