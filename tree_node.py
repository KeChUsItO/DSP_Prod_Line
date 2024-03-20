import pygame
import copy
import itertools
import math
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from gym import spaces
import os
import warnings
import pickle



import cloudpickle
import multiprocessing as mp

mp.connection.REDUCTION = cloudpickle

warnings.filterwarnings('ignore')
os.environ["TF_DEVICE_NAME"] = "/device:GPU:0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
FPS = 60
size = 1920, 1080
OFFSET_X = size[0] // 2
OFFSET_Y = size[1] // 2
GREEN = (0, 255, 0)
GR_BL = (33, 182, 168)
WHITE = (255, 255, 255)
M = 0.5
SCROLL_MULT = 0.1
MIN_SCROLL_MULT = 0.01
MAX_SCROLL_MULT = 0.1


def flatten_array(arr):
    flat_list = []
    for element in arr:
        if isinstance(element, list):
            flat_list.extend(flatten_array(element))
        else:
            flat_list.append(element)
    return flat_list


def normal_round(num, n_digits=0):
    if n_digits == 0:
        return int(num + 0.5)
    else:
        digit_value = 10 ** n_digits
        return int(num * digit_value + 0.5) / digit_value


class Game:
    MINING_SPEED = 1


class Item:
    def __init__(self, name, n, time=None, components=None, n_com=None, img_name=None, made_in=None, excesses=None):
        self.name = name
        self.n = n
        self.time = time
        self.components = components
        self.n_com = n_com
        self.made_in = made_in
        self.excesses = excesses
        if img_name:
            self.img_name = "Icons/" + img_name
        else:
            self.img_name = None


selection = "silicon_ore"


class Tree_node:
    def __init__(self, data, quantity=None, excess=None):
        self.children = []
        self.data = data
        self.quantity = copy.deepcopy(quantity) if quantity else None
        self.excess = copy.deepcopy(excess) if excess else None
        self.parent = None

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def tree_from_edge(self, edges):
        for edge in edges:
            if edge[0] == self.data:
                new_child = Tree_node(edge[1])
                new_child.tree_from_edge(edges)
                self.add_child(new_child)
        return self

    def add_quantity(self, name, quantity):
        if self.data == name:
            self.quantity = quantity
        else:
            for child in self.children:
                child.add_quantity(name, quantity)

    def add_excess(self, name, excess):
        if self.data == name:
            self.excess = excess
        else:
            for child in self.children:
                child.add_excess(name, excess)

    def draw(self, prefix='', is_last_child=False):
        print(prefix + ('└' if is_last_child else '├') + '── ' + str(self.data) + " " + str(self.quantity) + " " + str(
            self.excess))
        for i, child in enumerate(self.children):
            child.draw(prefix + ('    ' if is_last_child else '│   '), i == len(self.children) - 1)

    def potential_exchgn(self, init, search):
        result_array = []
        if search in self.data:
            result_array.append(self.data)
        if init != self.data:
            for child in self.children:
                if child.data != init:
                    for out in child.potential_exchgn(init, search):
                        result_array.append(out)
        return result_array

    def get_node_by_name(self, data):
        if self.data == data:
            return self
        else:
            for child in self.children:
                node = child.get_node_by_name(data)
                if node is not None:
                    return node
        return None

    def update_tree(self, sends, original_tree):
        if type(self.parent) == type(None):
            if self.data in sends.keys():
                self.quantity = original_tree.get_node_by_name(self.data).quantity - sum(sends[self.data])
            else:
                self.quantity = original_tree.get_node_by_name(self.data).quantity
        else:
            item = "_".join(self.parent.data.split("_")[:-1])
            item_act = "_".join(self.data.split("_")[:-1])
            if type(Graph.ITEMS[item].n) == list:
                items = []
                for child in self.parent.children:
                    items.append("_".join(child.data.split("_")[:-1]))
                for pos_i, pos in enumerate(Graph.ITEMS[item].components):
                    if check_same_items(items, pos):
                        posi_index = pos_i
                com_index = Graph.ITEMS[item].components[posi_index].index(item_act)
                if self.data in sends.keys():
                    self.quantity = self.parent.quantity * Graph.ITEMS[item].n_com[posi_index][com_index] / \
                                    Graph.ITEMS[item].n[posi_index] - sum(sends[self.data])
                else:
                    self.quantity = self.parent.quantity * Graph.ITEMS[item].n_com[posi_index][com_index] / \
                                    Graph.ITEMS[item].n[posi_index]
            else:
                com_index = Graph.ITEMS[item].components.index(item_act)
                if self.data in sends.keys():
                    self.quantity = self.parent.quantity * Graph.ITEMS[item].n_com[com_index] / Graph.ITEMS[
                        item].n - sum(sends[self.data])
                else:
                    self.quantity = self.parent.quantity * Graph.ITEMS[item].n_com[com_index] / Graph.ITEMS[item].n

        item = "_".join(self.data.split("_")[:-1])
        if type(Graph.ITEMS[item].n) == list:
            items = []
            for child in self.children:
                items.append("_".join(child.data.split("_")[:-1]))

            for pos_i, pos in enumerate(Graph.ITEMS[item].components):
                if check_same_items(items, pos):
                    posi_index = pos_i
            new_excs = []
            if Graph.ITEMS[item].excesses[posi_index]:
                for excs_num in Graph.ITEMS[item].excesses[posi_index][1]:
                    new_excs.append(excs_num / Graph.ITEMS[item].n[posi_index] * self.quantity)
                    self.excess = [Graph.ITEMS[item].excesses[posi_index][0], new_excs]
        else:
            new_excs = []
            if Graph.ITEMS[item].excesses:
                for excs_num in Graph.ITEMS[item].excesses[1]:
                    new_excs.append(excs_num / Graph.ITEMS[item].n * self.quantity)
                    self.excess = [Graph.ITEMS[item].excesses[0], new_excs]

        for child in self.children:
            child.update_tree(sends, original_tree)
        return

    def get_excess_dict(self):
        excess_dict = {}
        self._collect_excesses(excess_dict)
        return excess_dict

    def _collect_excesses(self, excess_dict):
        if self.excess is not None:
            excess_dict[self.data] = self.excess
        for child in self.children:
            child._collect_excesses(excess_dict)
    def sub_excesses(self, sends_excs):
        if self.data in sends_excs:
            for excess_ind, excess_num in enumerate(self.excess[1]):
                self.excess[1][excess_ind] -= sends_excs[self.data][1][excess_ind]

        for child in self.children:
            child.sub_excesses(sends_excs)

    def reward(self):
        exc_num_act = 0
        if self.quantity < 0:
            exc_num_act += abs(self.quantity)
        if self.excess:
            for exc_qtty in self.excess[1]:
                exc_num_act += abs(exc_qtty)
        for child in self.children:
            exc_num_act += child.reward()
        return exc_num_act

    def observe(self, obs=None):
        if not obs:
            obs = []
        if self.excess:
            for exc_qtty in self.excess[1]:
                obs.append(exc_qtty)
        if self.children:
            for child in self.children:
                ret = child.observe()
                for ret_num in ret:
                    obs.append(ret_num)

        return obs


def isListEmpty(inList):
    if isinstance(inList, list):  # Is a list
        return all(map(isListEmpty, inList))
    return False  # Not a list


def check_same_items(arr1, arr2):
    # Create sets of the arrays to remove duplicates
    set1 = set(arr1)
    set2 = set(arr2)

    # Check if the sets are the same
    return set1 == set2


def my_ceil(a, precision=0):
    return np.true_divide(np.ceil(a * 10 ** precision), 10 ** precision)


class Excess_manager(gym.Env):
    def __init__(self, n_acts, n_obs, tree, tree_copy, dict_potential, excesses, dict_excesses, relation_dict,
                 per_min_original):
        self.original_tree = tree_copy
        self.tree = tree
        self.excesses = excesses
        self.per_min_original = per_min_original
        self.dict_potential = dict_potential
        self.relation_dict = relation_dict
        self.dict_excesses = dict_excesses
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1, n_acts), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=0, shape=(1,), dtype=np.float32)
        self.n_obs = n_obs
        self.factories_original = line.get_factories(per_min_original)
        self.count_original = line.count_factories(self.factories_original)
        self.reset()

    def reset(self):
        self.tree = copy.deepcopy(self.original_tree)
        obs = [0]
        return obs

    def step(self, action):
        action = action[0]
        info = {}
        done = False
        sends = {}
        for act_i, act in enumerate(action):
            for key in self.relation_dict:
                for indx in self.relation_dict[key]:
                    if indx == act_i:
                        excs_ind = self.tree.get_node_by_name(key).excess[0].index(
                            "_".join(self.dict_potential[act_i].split("_")[:-1]))
                        send = math.ceil(self.original_tree.get_node_by_name(key).excess[1][excs_ind] * act)
                        if key not in sends.keys():
                            sends[key] = []
                        sends[key].append(send)
        sends2 = {}
        for key, value in self.dict_potential.items():
            for key2, value2 in self.relation_dict.items():
                for send_value_i, send_value in enumerate(value2):
                    if send_value == key:
                        if value not in sends2.keys():
                            sends2[value] = []
                        sends2[value].append(sends[key2][send_value_i])
        sends_excs = {}

        for sends_excess in self.excesses:
            for indexes_i, indexes in enumerate(self.relation_dict[sends_excess]):
                if not sends_excess in sends_excs.keys():
                    sends_excs[sends_excess] = [self.excesses[sends_excess][0],
                                                [0 for _ in self.excesses[sends_excess][0]]]

                item = "_".join(self.dict_potential[indexes].split("_")[:-1])

                sub_index = self.excesses[sends_excess][0].index(item)
                sends_excs[sends_excess][1][sub_index] += sends[sends_excess][indexes_i]

        self.tree.update_tree(sends2, self.original_tree)
        self.tree.sub_excesses(sends_excs)
        self.sends = sends2
        info["observation"] = self.tree.observe()

        obs = [0]
        per_min = {}
        for key in self.per_min_original:
            per_min[key] = self.tree.get_node_by_name(key).quantity
        factories = line.get_factories(per_min)
        count = line.count_factories(factories)
        fact_array = line.factories_array(factories)
        if min(info["observation"]) < 0:
            reward = -10
        else:
            reward = 10 * np.e ** -(count / self.count_original)
        info["tree"] = copy.deepcopy(self.tree)
        info["sends2"] = copy.deepcopy(sends2)
        info["sends"] = copy.deepcopy(sends)
        return obs, reward, done, info

    def render(self, mode='human'):
        self.tree.draw()

    def return_tree(self):
        return self.tree


class Graph:
    MAX_ASSEMBLER_TIER = 3
    MAX_SMELTING_TIER = 2
    MAX_MINING_TIER = 2
    MAX_CHEMICAL_TIER = 2
    not_able = []
    factory_images = {"assembler": ["Icons/Icon_Assembling_Machine_Mk.I.png", "Icons/Icon_Assembling_Machine_Mk.II.png",
                                    "Icons/Icon_Assembling_Machine_Mk.III.png"],
                      "smelting_facility": ["Icons/Icon_Arc_Smelter.png", "Icons/Icon_Plane_Smelter.png"],
                      "mining_machine": ["Icons/Icon_Mining_Machine.png", "Icons/Icon_Advanced_Mining_Machine.png"],
                      "chemical_facility": ["Icons/Icon_Chemical_Plant.png", "Icons/Icon_Quantum_Chemical_Plant.png"],
                      "research_facility": ["Icons/Icon_Matrix_Lab.png"],
                      "refining_facility": ["Icons/Icon_Oil_Refinery.png"],
                      "oil_extraction_facility": ["Icons/Icon_Oil_Extractor.png"],
                      "water_pumping_facility": ["Icons/Icon_Water_Pump.png"], }
    ITEMS = {
        "gear": Item("gear", 1, 1, ["iron_ingot"], [1], "Icon_Gear.png", "assembler"),
        "iron_ingot": Item("iron_ingot", 1, 1, ["iron_ore"], [1], "Icon_Iron_Ingot.png", "smelting_facility"),
        "glass": Item("glass", 1, 2, ["stone"], [2], "Icon_Glass.png", "smelting_facility"),
        "stone_brick": Item("stone_brick", 1, 1, ["stone"], [1], "Icon_Stone_Brick.png", "smelting_facility"),
        "stone": Item("stone", 1, 2, ["stone_ore_vein"], [1], "Icon_Stone.png", "mining_machine"),
        "iron_ore": Item("iron_ore", 1, 2, ["iron_ore_vein"], [1], "Icon_Iron_Ore.png", "mining_machine"),
        "stone_ore_vein": Item("stone_ore_vein", 0, 0, None, None, "Icon_Stone_Ore_Vein.png"),
        "iron_ore_vein": Item("iron_ore_vein", 0, 0, None, None, "Icon_Iron_Ore_Vein.png"),
        "silicon_ore_vein": Item("silicon_ore_vein", 0, 0, None, None, "Icon_Silicon_Ore_Vein.png"),
        "copper_ore_vein": Item("copper_ore_vein", 0, 0, None, None, "Icon_Copper_Vein.png"),
        "copper_ore": Item("copper_ore", 1, 2, ["copper_ore_vein"], [1], "Icon_Copper_Ore.png", "mining_machine"),
        "coal": Item("coal", 1, 2, ["coal_vein"], [1], "Icon_Coal.png", "mining_machine"),
        "coal_vein": Item("coal_vein", 0, 0, None, None, "Icon_Coal_Vein.png"),
        "magnet": Item("magnet", 1, 1.5, ["iron_ore"], [1], "Icon_Magnet.png", "smelting_facility"),
        "copper_ingot": Item("copper_ingot", 1, 1, ["copper_ore"], [1], "Icon_Copper_Ingot.png", "smelting_facility"),
        "magnetic_coil": Item("magnetic_coil", 2, 1, ["magnet", "copper_ingot"], [2, 1], "Icon_Magnetic_Coil.png",
                              "assembler"),

        "conveyor_belt_mk.I": Item("conveyor_belt_mk.I", 3, 1, ["iron_ingot", "circuit_board"], [1, 1],
                                   "Icon_Sorter_Mk.I.png",
                                   "assembler"),
        "sorter_mk.I": Item("sorter_mk.I", 1, 1, ["iron_ingot", "gear"], [2, 1],
                            "Icon_Conveyor_Belt_Mk.I.png",
                            "assembler"),
        "storage.I": Item("storage.I", 1, 2, ["iron_ingot", "stone_brick"], [4, 4],
                          "Icon_Storage_Mk.I.png",
                          "assembler"),
        "electric_motor": Item("electric_motor", 1, 2, ["iron_ingot", "gear", "magnetic_coil"], [2, 1, 1],
                               "Icon_Electric_Motor.png", "assembler"),
        "wind_turbine": Item("wind_turbine", 1, 4, ["iron_ingot", "gear", "magnetic_coil"], [6, 1, 3],
                             "Icon_Wind_Turbine.png", "assembler"),
        "assembling_machine_mk.I": Item("assembling_machine_mk.I", 1, 2, ["iron_ingot", "gear", "circuit_board"],
                                        [4, 8, 4],
                                        "Icon_Assembling_Machine_Mk.I.png", "assembler"),
        "tesla_tower": Item("tesla_tower", 1, 1, ["iron_ingot", "magnetic_coil"], [1, 1],
                            "Icon_Tesla_Tower.png", "assembler"),
        "circuit_board": Item("circuit_board", 2, 1, ["iron_ingot", "copper_ingot"],
                              [2, 1],
                              "Icon_Circuit_Board.png", "assembler"),
        "electromagnetic_matrix": Item("electromagnetic_matrix", 1, 3, ["magnetic_coil", "circuit_board"],
                                       [1, 1],
                                       "Icon_Electromagnetic_Matrix.png", "research_facility"),
        "mining_machine": Item("mining_machine", 1, 3, ["iron_ingot", "circuit_board", "magnetic_coil", "gear"],
                               [4, 2, 2, 2],
                               "Icon_Mining_Machine.png", "assembler"),
        "arc_smelter": Item("arc_smelter", 1, 3, ["iron_ingot", "stone_brick", "circuit_board", "magnetic_coil"],
                            [4, 2, 4, 2],
                            "Icon_Arc_Smelter.png", "assembler"),
        "matrix_lab": Item("matrix_lab", 1, 3, ["iron_ingot", "glass", "circuit_board", "magnetic_coil"],
                           [8, 4, 4, 4],
                           "Icon_Matrix_Lab.png", "assembler"),
        "storage_tank": Item("storage_tank", 1, 2, ["iron_ingot", "stone_brick", "glass"],
                             [8, 4, 4],
                             "Icon_Storage_Tank.png", "assembler"),
        "water_pump": Item("water_pump", 1, 4, ["iron_ingot", "stone_brick", "electric_motor", "circuit_board"],
                           [8, 4, 4, 2], "Icon_Water_Pump.png", "assembler"),
        "prism": Item("prism", 2, 2, ["glass"],
                      [3], "Icon_Prism.png", "assembler"),
        "plasma_exciter": Item("plasma_exciter", 1, 2, ["magnetic_coil", "prism"],
                               [4, 2], "Icon_Plasma_Exciter.png", "assembler"),
        "wireless_power_tower": Item("wireless_power_tower", 1, 3, ["tesla_tower", "plasma_exciter"],
                                     [1, 3], "Icon_Wireless_Power_Tower.png", "assembler"),
        "splitter": Item("splitter", 1, 2, ["iron_ingot", "gear", "circuit_board"],
                         [3, 2, 1], "Icon_Splitter.png", "assembler"),
        "sorter_mk.II": Item("sorter_mk.II", 2, 1, ["sorter_mk.I", "electric_motor"],
                             [2, 1], "Icon_Sorter_Mk.II.png", "assembler"),
        "traffic_monitor": Item("traffic_monitor", 1, 2, ["iron_ingot", "gear", "glass", "circuit_board"],
                                [3, 2, 1, 2], "Icon_Traffic_Monitor.png", "assembler"),
        "steel": Item("steel", 1, 3, ["iron_ingot"],
                      [3], "Icon_Steel.png", "smelting_facility"),
        "silicon_ore": Item("silicon_ore", [1, 1], [10, 2], [["stone"], ["silicon_ore_vein"]],
                            [[10], [1]], "Icon_Silicon_Ore.png", ["smelting_facility", "mining_machine"]),
        "high-purity_silicon": Item("high-purity_silicon", 1, 2, ["silicon_ore"], [2], "Icon_High-Purity_Silicon.png",
                                    "smelting_facility"),
        "energetic_graphite": Item("energetic_graphite", [1, 1], [2, 4], [["coal"], ["refined_oil", "hydrogen"]],
                                   [[2], [1, 2]], "Icon_Energetic_Graphite.png",
                                   ["smelting_facility", "refining_facility"], [[[], []], [["hydrogen"], [3]]]),
        "refined_oil": Item("refined_oil", [2, 3], [4, 4], [["crude_oil"], ["refined_oil", "hydrogen", "coal"]],
                            [[2], [2, 1, 1]], "Icon_Refined_Oil.png",
                            ["refining_facility", "refining_facility"], [[["hydrogen"], [1]], [[], []]]),
        "crude_oil": Item("crude_oil", 1, 1, ["crude_oil_vein"],
                          [1], "Icon_Crude_Oil.png",
                          "oil_extraction_facility"),
        "crude_oil_vein": Item("crude_oil_vein", 0, 0, None, None, "Icon_Crude_Oil_Vein.png"),
        "hydrogen": Item("hydrogen", [1, 1, 3], [4, 2, 4], [["crude_oil"], ["fire_ice"], ["refined_oil", "hydrogen"]],
                         [[2], [2],
                          [1, 2]], "Icon_Hydrogen.png",
                         ["refining_facility", "chemical_facility", "refining_facility"],
                         [[["refined_oil"], [2]], [["graphene"], [2]], [["energetic_graphite"], [1]]]),
        "fire_ice": Item("fire_ice", 1, 2, ["fire_ice_vein"],
                         [1], "Icon_Fire_Ice.png",
                         "mining_machine"),
        "fire_ice_vein": Item("fire_ice_vein", 0, 0, None, None, "Icon_Fire_Ice_Vein.png"),
        "graphene": Item("graphene", [2, 2], [3, 2], [["energetic_graphite", "sulfuric_acid"], ["fire_ice"]],
                         [[3, 1], [2]], "Icon_Graphene.png",
                         ["chemical_facility", "chemical_facility"], [[[], []], [["hydrogen"], [1]]]),
        "sulfuric_acid": Item("sulfuric_acid", [4, 1], [6, 1.2],
                              [["refined_oil", "stone", "water"], ["sulfuric_acid_ocean"]],
                              [[6, 8, 4], [1]], "Icon_Sulfuric_Acid.png",
                              ["chemical_facility", "water_pumping_facility"], [[[], []], [[], []]]),
        "water": Item("water", 1, 1.2, ["water_ocean"],
                      [1], "Icon_Water.png",
                      "water_pumping_facility"),
        "water_ocean": Item("water_ocean", 0, 0, None, None, "Icon_Water.png"),
        "sulfuric_acid_ocean": Item("sulfuric_acid_ocean", 0, 0, None, None, "Icon_Sulfuric_Acid.png"),
        "energy_matrix": Item("energy_matrix", 1, 6, ["energetic_graphite", "hydrogen"], [2, 2],
                              "Icon_Energy_Matrix.png", "research_facility"),
        "solar_panel": Item("solar_panel", 1, 6, ["copper_ingot", "high-purity_silicon", "circuit_board"], [10, 10, 5],
                            "Icon_Solar_Panel.png",
                            "assembler"),
    }

    def __init__(self, objective, quantity):
        self.po = None
        self.code = {}
        self.objective = objective
        self.quantity = quantity

    def calculate(self, ind=0):
        self.new_edges = []
        self.ind = ind
        self.per_min_excs = []
        self.paths, self.layers, self.heights = self.get_paths(ind=ind)
        self.vertexes = self.get_vertexes()
        if self.check_edge():
            self.get_numbers()
            self.colour_edges()
            self.factories = self.get_factories(self.per_min)
            self.factories2 = self.get_factories(self.per_min_2)
            print("Factories", self.count_factories(self.factories), self.count_factories(self.factories2))

    def get_vertexes(self):
        vertexes = []
        for path in self.paths:
            if path not in vertexes:
                vertexes.append(path)
            for v in self.paths[path]:
                if v not in vertexes:
                    vertexes.append(v)
        return vertexes

    def check_edge(self):
        for va, vb in self.edges[1:]:
            for not_allowed in self.not_able:
                if not_allowed in va or not_allowed in vb:
                    return False
        return True

    def find_edges(self, prev_name=None, prev_path=None, counts=None):
        if prev_path is None:
            prev_path = []
        if counts is None:
            counts = {}
        if Graph.ITEMS[self.objective].name not in list(counts.keys()):
            counts[Graph.ITEMS[self.objective].name] = 0
        else:
            counts[Graph.ITEMS[self.objective].name] += 1
        name_ind = Graph.ITEMS[self.objective].name + f"_{counts[Graph.ITEMS[self.objective].name]}"

        prev_path.append((prev_name, name_ind))
        if Graph.ITEMS[self.objective].components:
            for n_comp, comp in enumerate(Graph.ITEMS[self.objective].components):
                prev_path = Graph(comp, 1).find_edges(name_ind, prev_path, counts)
        return prev_path

    def count_factories(self, factories):
        total_fact = 0
        for key, item in factories.items():
            key_clean = "_".join(key.split("_")[:-1])
            if Graph.ITEMS[key_clean].n_com:
                if "mining_machine" not in item[0] and "oil_extraction_facility" not in item[
                    0] and "water_pumping_facility" not in item[0]:
                    for fact in item[1]:
                        total_fact += fact
        return total_fact

    def factories_array(self, factories):
        fact_array = []
        for key, item in factories.items():
            key_clean = "_".join(key.split("_")[:-1])
            if Graph.ITEMS[key_clean].n_com:
                if "mining_machine" not in item[0] and "oil_extraction_facility" not in item[
                    0] and "water_pumping_facility" not in item[0]:
                    for fact in item[1]:
                        fact_array.append(fact)
        return fact_array

    def find_edges_full(self, ind=0):
        def get_combination(needed, parent, repeated=[]):
            for_mixing = []
            for itms in needed:
                if len(itms) > 0:
                    for itm in itms:
                        if Graph.ITEMS[itm].components:
                            if type(Graph.ITEMS[itm].components[0]) != str:
                                components_copy = [comp for comp in Graph.ITEMS[itm].components if
                                                   not any(map(lambda each: each in comp, repeated))]
                                for_mixing.append(components_copy)
                            else:
                                for_mixing.append([Graph.ITEMS[itm].components])
                        else:
                            for_mixing.append([[]])
                else:
                    for_mixing.append([[]])

            new_combinations = list(itertools.product(*for_mixing))
            if not isListEmpty(for_mixing):
                for n_comb in new_combinations:
                    repeated = []

                    new_child = Tree_node(list(n_comb))
                    parent.add_child(new_child)
                    data_copy = []
                    for point_data_i, point_data in enumerate(new_child.parent.data):
                        if point_data == []:
                            data_copy.append(['1'])
                        else:
                            data_copy.append(point_data)
                    for compo_i, compo in enumerate(list(itertools.chain.from_iterable(data_copy))):
                        if compo != '1':
                            # print(compo, new_child.data[compo_i])
                            pass
                        if compo in new_child.data[compo_i]:
                            repeated.append(compo)

                    get_combination(list(n_comb), new_child, repeated)

        def get_all_paths(node):
            if len(node.children) == 0:
                return [[node.data]]
            else:
                return [
                    [node.data] + path for child in node.children for path in get_all_paths(child)
                ]

        def get_edge_from_comb(comb):
            prev = []
            edges = []
            seen = {}
            comb_copy = []
            for c_i, c in enumerate(comb):

                c_cop = []
                for row_i, row in enumerate(c):
                    row_cop = []
                    for item_i, item in enumerate(row):
                        if item not in seen.keys():
                            seen[item] = 0
                        else:
                            seen[item] += 1
                        row_cop.append(item + f"_{seen[item]}")
                    c_cop.append(row_cop)
                comb_copy.append(c_cop)
            comb = comb_copy
            for c_i, c in enumerate(comb):
                for row_i, row in enumerate(c):
                    for item_i, item in enumerate(row):
                        if prev:
                            edges.append((prev[row_i], item))
                        else:
                            edges.append((None, item))

                prev = []
                for row_i, row in enumerate(c):

                    if row == []:
                        prev.append('')
                    else:
                        for item in row:
                            prev.append(item)

            return edges

        root = Tree_node([[self.objective]])

        get_combination([[self.objective]], root)
        self.Tree = root
        self.all_posibilities = len(get_all_paths(root))
        return get_edge_from_comb(get_all_paths(root)[ind])

    def layer(self, objs):
        m_arrs = []
        for prev in objs:
            arrs = []
            for n in prev:
                n_a = []
                for obj in n:
                    if Graph.ITEMS[obj].components:
                        if type(Graph.ITEMS[obj].name) == list:
                            for i, nam in enumerate(Graph.ITEMS[obj].name):
                                arr = []
                                for j, comp in enumerate(Graph.ITEMS[obj].components[i]):
                                    arr.append(comp)
                                arrs.append(arr)
                        else:
                            arr = []
                            for j, comp in enumerate(Graph.ITEMS[obj].components):
                                arr.append(comp)

                    n_a.append(arr)
                arrs.append(n_a)
            m_arrs.append(arrs)
        return arrs

    def get_paths(self, ind=0):
        path = {}
        path_nums = {"None": -1}
        self.edges = self.find_edges_full(ind=ind)
        for edge in self.edges:
            path_nums[str(edge[1])] = path_nums[str(edge[0])] + 1

            if edge[0]:
                if edge[0] not in path.keys():
                    path[str(edge[0])] = [edge[1]]

                else:
                    path[str(edge[0])].append(edge[1])
        layers = {val: [key for key, value in path_nums.items() if value == val] for val in set(path_nums.values())}
        heights = {}
        for lay in layers.values():
            for i, item in enumerate(lay):
                heights[item] = i - len(lay) // 2

        return path, layers, heights

    def get_numbers(self):
        print("______________Numbers______________")
        per_min = {self.objective + "_0": self.quantity}
        excesses_graphs = {}
        excesses = {}
        fact_tree = Tree_node(self.objective + "_0", self.quantity)
        fact_tree.tree_from_edge(self.edges[1:])

        for edge in self.edges[1:]:
            item = "_".join(edge[0].split("_")[:-1])
            item_2 = "_".join(edge[1].split("_")[:-1])

            if type(Graph.ITEMS[item].components[0]) == str:
                component = "_".join(edge[1].split("_")[:-1])

                component_ind = Graph.ITEMS[item].components.index(component)
                per_min[edge[1]] = per_min[edge[0]] / Graph.ITEMS[item].n * Graph.ITEMS[item].n_com[component_ind]
            else:
                sub_ncom = []
                for edge2 in self.edges:
                    if edge2[0] == edge[0]:
                        sub_ncom.append("_".join(edge2[1].split("_")[:-1]))
                component = "_".join(edge[1].split("_")[:-1])
                n_com_ind = Graph.ITEMS[item].components.index(sub_ncom)
                component_ind = Graph.ITEMS[item].components[n_com_ind].index(component)
                per_min[edge[1]] = per_min[edge[0]] / Graph.ITEMS[item].n[n_com_ind] * \
                                   Graph.ITEMS[item].n_com[n_com_ind][component_ind]

                for excess_name_i, excess_name in enumerate(Graph.ITEMS[item].excesses[n_com_ind][0]):
                    if excess_name not in excesses_graphs.keys():
                        excesses_graphs[excess_name] = np.zeros((len(self.vertexes), len(self.vertexes)))
                    excesses_graphs[excess_name][self.vertexes.index(edge[0])][self.vertexes.index(edge[0])] = \
                        Graph.ITEMS[item].excesses[n_com_ind][1][excess_name_i] * per_min[
                            edge[0]] / Graph.ITEMS[item].n[n_com_ind]
                    excesses[edge[0]] = [Graph.ITEMS[item].excesses[n_com_ind][0], [exc * per_min[
                        edge[0]] / Graph.ITEMS[item].n[n_com_ind] for exc in Graph.ITEMS[item].excesses[n_com_ind][1]]]

            fact_tree.add_quantity(edge[1], per_min[edge[1]])
            if edge[0] in excesses.keys():
                fact_tree.add_excess(edge[0], excesses[edge[0]])
            if item_2 == "crude_oil_vein":
                per_min[edge[1]] = my_ceil(
                    per_min[edge[0]] / (Graph.ITEMS[item].n * Game.MINING_SPEED) * Graph.ITEMS[item].n_com[
                        component_ind], 2)
        self.per_min = per_min
        potential = {}
        for key in excesses.keys():
            if key not in potential.keys():
                potential[key] = []
            for search in excesses[key][0]:
                potential[key].append(fact_tree.potential_exchgn(key, search))
        n_acts = 0
        n_obs = 0
        dict_potential = {}
        dict_excesses = {}
        for sub_potential in potential:
            if potential[sub_potential] != [[]]:
                dict_excesses[n_obs] = sub_potential
                for prod in potential[sub_potential]:
                    for sub_prod in prod:
                        dict_potential[n_acts] = sub_prod
                        n_acts += 1

                n_obs += 1

        for key in potential.copy().keys():
            if potential[key] == [[]]:
                potential.pop(key)
                excesses.pop(key)

        aux_ind = 0
        relation_dict = {}
        for key in potential:
            relation_dict[key] = []
            for comp in potential[key][0]:
                relation_dict[key].append(aux_ind)
                aux_ind += 1

        fact_tree.draw()
        original_tree = copy.deepcopy(fact_tree)
        env = Excess_manager(n_acts, n_obs, fact_tree, original_tree, dict_potential, excesses, dict_excesses,
                             relation_dict, self.per_min)
        besttree = None
        best_action = None
        best_send2 = None
        best_send = None
        bestreward = -10

        def make_env():
            return Excess_manager(n_acts, n_obs, fact_tree, original_tree, dict_potential, excesses, dict_excesses,
                                  relation_dict, self.per_min)

        if n_obs > 0:
            env = SubprocVecEnv([make_env for _ in range(4)])
            model = PPO("MlpPolicy", env, verbose=1)
            obs = env.reset()
            for i in range(5):
                action, _states = model.predict(obs)
                print(action)
                obs, rewards, done, info = env.step(action)

                if done:
                    break
            print(obs, rewards, done, action, info)
            env.close()
            rewstd = -1
            rewmean = -10
            while rewstd > 10 ** -6 or np.round(np.mean(rewmean), 1) == -10:
                model.learn(total_timesteps=10_000)

                print("________")
                obs = env.reset()
                rewmean = []
                trees = []
                actions = []
                sends2 = []
                sends = []

                for i in range(30):
                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)
                    rewmean.append(rewards[0])
                    trees.append(info[0]["tree"])
                    sends2.append(info[0]["sends2"])
                    sends.append(info[0]["sends"])
                    actions.append(action)

                rewstd = np.std(rewmean)
                if max(rewmean) > bestreward:
                    bestreward = max(rewmean)
                    besttree = copy.deepcopy(trees[np.argmax(np.array(rewmean))])
                    best_send2 = copy.deepcopy(sends2[np.argmax(np.array(rewmean))])
                    best_send = copy.deepcopy(sends[np.argmax(np.array(rewmean))])
                    best_action = copy.deepcopy(actions[np.argmax(np.array(rewmean))])
                print("MEAN", np.round(np.mean(rewmean), 1), "+/-", rewstd,
                      np.round(rewstd / np.mean(rewmean), 2))
            print("BESTTREE")
            besttree.draw()

            print("____________________-")
            env.close()

            print(best_action)
            print(dict_potential)
            print(potential)
            print(excesses)
            print(relation_dict)
            print(dict_excesses)
            print(best_send2)
            print(best_send)
        self.per_min_2 = {}

        if besttree:
            for key in self.per_min:
                self.per_min_2[key] = besttree.get_node_by_name(key).quantity

        if type(best_action) != type(None):
            self.new_edges = []
            self.new_edges_back = []
            self.per_min_excs = []

            for b_act_i, b_act in enumerate(best_action[0][0]):
                if b_act != 0:
                    dict_i = 0
                    for keys, values in relation_dict.items():
                        for sub_indx_i, sub_indx in enumerate(values):
                            if sub_indx == b_act_i:
                                come_key = keys
                                # self.per_min_excs.append(list(best_send.values())[b_act_i])
                                break
                        dict_i += 1

                    self.per_min_excs.append(best_send[come_key][potential[come_key][0].index(dict_potential[b_act_i])])
                    if besttree.get_node_by_name(dict_potential[b_act_i]).parent:
                        new_edge = (come_key, besttree.get_node_by_name(dict_potential[b_act_i]).parent.data)
                    else:
                        new_edge = (come_key, besttree.get_node_by_name(dict_potential[b_act_i]).data)
                    new_edge_back = (come_key, besttree.get_node_by_name(dict_potential[b_act_i]).data)
                    self.new_edges.append(new_edge)
                    self.new_edges_back.append(new_edge_back)
                    print("New edge back", self.new_edges_back)
                    print(self.per_min_excs)
        else:
            self.per_min_2 = per_min

        self.excesses_graphs = excesses_graphs
        self.excesses = excesses
        print(self.new_edges)
        print(best_action)

        print("______________END______________")

    def colour_edges(self):
        edge_colours = {}
        for key, value in self.per_min.items():
            for edge in self.edges:
                if edge[1] == key:
                    if value <= 6 * 60:
                        edge_colours[edge] = ORANGE
                    elif 6 * 60 < value <= 12 * 60:
                        edge_colours[edge] = GR_BL
                    else:
                        edge_colours[edge] = BLUE
        self.edge_colours = edge_colours
        if self.per_min_2:
            edge_colours = {}
            for key, value in self.per_min_2.items():
                for edge in self.edges:
                    if edge[1] == key:
                        if value <= 6 * 60:
                            edge_colours[edge] = ORANGE
                        elif 6 * 60 < value <= 12 * 60:
                            edge_colours[edge] = GR_BL
                        else:
                            edge_colours[edge] = BLUE
        self.edge_colours_2 = edge_colours
        if self.per_min_excs:
            edge_colours = []
            for edge_i, edge in enumerate(self.new_edges):
                value = self.per_min_excs[edge_i]
                if value <= 6 * 60:
                    edge_colours.append(ORANGE)
                elif 6 * 60 < value <= 12 * 60:
                    edge_colours.append(GR_BL)
                else:
                    edge_colours.append(BLUE)
            self.edge_colours_excess = edge_colours

    def get_factories(self, per_min):
        factories_get = {}
        for key, val in per_min.items():
            key_clean = "_".join(key.split("_")[:-1])
            if type(Graph.ITEMS[key_clean].time) != float and type(Graph.ITEMS[key_clean].time) != int:
                factories = None
                sub_ncom = []
                for edge2 in self.edges:
                    if edge2[0] == key:
                        sub_ncom.append("_".join(edge2[1].split("_")[:-1]))

                n_com_ind = Graph.ITEMS[key_clean].components.index(sub_ncom)

                time = Graph.ITEMS[key_clean].time[n_com_ind]
                number = Graph.ITEMS[key_clean].n[n_com_ind]
                machine = Graph.ITEMS[key_clean].made_in[n_com_ind]

            else:
                time = Graph.ITEMS[key_clean].time
                number = Graph.ITEMS[key_clean].n
                machine = Graph.ITEMS[key_clean].made_in

            if machine == "assembler":
                prod_1 = 60 / time * number
                prods = [prod_1 * 0.75, prod_1, prod_1 * 1.5]
                if Graph.MAX_ASSEMBLER_TIER == 1:
                    num_ass = val / prods[0]
                    if num_ass % 1 != 0:
                        factories = ["assembler", [int(num_ass) + 1, 0, 0]]
                    else:
                        factories = ["assembler", [int(num_ass), 0, 0]]
                elif Graph.MAX_ASSEMBLER_TIER == 2:
                    num_ass = val / prods[1]
                    if num_ass % 1 != 0:

                        if val / num_ass * (num_ass - int(num_ass)) <= prods[0]:
                            factories = ["assembler", [1, int(num_ass), 0]]
                        else:
                            factories = ["assembler", [0, int(num_ass) + 1, 0]]

                    else:
                        factories = ["assembler", [0, int(num_ass), 0]]

                elif Graph.MAX_ASSEMBLER_TIER == 3:
                    num_ass = val / prods[2]
                    if num_ass % 1 != 0:
                        extra = val / num_ass * (num_ass - int(num_ass))
                        if extra <= prods[0]:
                            factories = ["assembler", [1, 0, int(num_ass)]]
                        elif prods[0] < extra <= prods[1]:
                            factories = ["assembler", [0, 1, int(num_ass)]]
                        else:
                            factories = ["assembler", [0, 0, int(num_ass) + 1]]


                    else:
                        factories = ["assembler", [0, 0, int(num_ass)]]

            elif machine == "smelting_facility":
                prod_1 = 60 / time * number
                prods = [prod_1, prod_1 * 2]
                if Graph.MAX_SMELTING_TIER == 1:
                    num_ass = val / prods[0]
                    if num_ass % 1 != 0:
                        factories = ["smelting_facility", [int(num_ass) + 1, 0, 0]]
                    else:
                        factories = ["smelting_facility", [int(num_ass), 0, 0]]
                elif Graph.MAX_SMELTING_TIER == 2:
                    num_ass = val / prods[1]
                    if num_ass % 1 != 0:

                        if val / num_ass * (num_ass - int(num_ass)) <= prods[0]:
                            factories = ["smelting_facility", [1, int(num_ass), 0]]
                        else:
                            factories = ["smelting_facility", [0, int(num_ass) + 1, 0]]

                    else:
                        factories = ["smelting_facility", [0, int(num_ass), 0]]

            elif machine == "mining_machine":
                prod_1 = 60 / time * number
                prods = [prod_1, prod_1 * 2]
                if Graph.MAX_MINING_TIER == 1:
                    num_ass = val / prods[0]
                    if num_ass % 1 != 0:
                        factories = ["mining_machine", [int(num_ass) + 1, 0]]
                    else:
                        factories = ["mining_machine", [int(num_ass), 0]]
                elif Graph.MAX_MINING_TIER == 2:
                    num_ass = val / prods[1]
                    if num_ass % 1 != 0:
                        factories = ["mining_machine", [0, int(num_ass) + 1]]
                    else:
                        factories = ["mining_machine", [0, int(num_ass)]]

            elif machine == "oil_extraction_facility":
                prod_1 = 60 / time * number
                prods = [prod_1]
                num_ass = val / prods[0]
                if num_ass != 0:
                    factories = ["oil_extraction_facility", [1]]
                else:
                    factories = ["oil_extraction_facility", [0]]
            elif machine == "research_facility":
                prod_1 = 60 / time * number
                prods = [prod_1]
                num_ass = val / prods[0]
                if num_ass % 1 != 0:
                    factories = ["research_facility", [int(num_ass) + 1]]
                else:
                    factories = ["research_facility", [int(num_ass)]]

            elif machine == "refining_facility":
                prod_1 = 60 / time * number
                prods = [prod_1]
                num_ass = val / prods[0]
                if num_ass % 1 != 0:
                    factories = ["refining_facility", [int(num_ass) + 1]]
                else:
                    factories = ["refining_facility", [int(num_ass)]]

            elif machine == "chemical_facility":
                prod_1 = 60 / time * number
                prods = [prod_1, prod_1 * 2]
                if Graph.MAX_CHEMICAL_TIER == 1:
                    num_ass = val / prods[0]
                    if num_ass % 1 != 0:
                        factories = ["chemical_facility", [int(num_ass) + 1, 0]]
                    else:
                        factories = ["chemical_facility", [int(num_ass), 0]]
                elif Graph.MAX_CHEMICAL_TIER == 2:
                    num_ass = val / prods[1]
                    if num_ass % 1 != 0:
                        factories = ["chemical_facility", [0, int(num_ass) + 1]]
                    else:
                        factories = ["chemical_facility", [0, int(num_ass)]]

            elif machine == "water_pumping_facility":
                prod_1 = 60 / time * number
                prods = [prod_1]
                num_ass = val / prods[0]
                if num_ass % 1 != 0:
                    factories = ["water_pumping_facility", [int(num_ass) + 1]]
                else:
                    factories = ["water_pumping_facility", [int(num_ass)]]
            factories_get[key] = factories
        return factories_get

    def draw(self):
        global OFFSET_X
        global OFFSET_Y
        global M
        global SCROLL_MULT
        global MIN_SCROLL_MULT
        paths_num = {}
        ANGLE = 45
        LEN = 500
        LEN_DECAY_MULT = 0.9
        positions = {}
        self.per_min_act = self.per_min
        self.edge_colours_act = self.edge_colours
        self.factories_act = self.factories

        for p, key in enumerate(self.paths.keys()):
            lines = len(self.paths[key])
            if lines % 2 == 0:
                angles = [ANGLE // (i + 1) for i in range(lines // 2)]
                angles += [-an for an in angles]

            else:
                angles = [ANGLE // (i + 1) for i in range(lines // 2)]
                angles += [-an for an in angles]
                angles += [0]
            angles = sorted(angles)
            x_dif = 0

            if p == 0:
                positions[key] = (0, 0)
            for i, comp in enumerate(self.paths[key]):
                if comp not in positions:
                    for lay in self.layers:
                        if comp in self.layers[lay]:
                            x_dif = lay

                    positions[comp] = (LEN * -x_dif,
                                       self.heights[comp] * LEN)

        pygame.init()
        screen = pygame.display.set_mode(size)
        pygame.display.flip()
        run = True
        dragging = False
        clock = pygame.time.Clock()
        mode = "normal"

        while run:
            clock.tick(FPS)
            screen.fill(BLACK)
            for event in pygame.event.get():
                if event.type == pygame.QUIT: run = False

                if event.type == pygame.MOUSEBUTTONDOWN:
                    dragging = True
                    first_drag = True
                elif event.type == pygame.MOUSEBUTTONUP:
                    dragging = False
                if event.type == pygame.MOUSEWHEEL:
                    M += event.y * SCROLL_MULT
                    if M <= 0.01:
                        M = 0.01
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_o:
                        self.per_min_act = copy.deepcopy(self.per_min_2)
                        self.factories_act = self.factories2
                        if self.edge_colours_2:
                            self.edge_colours_act = copy.deepcopy(self.edge_colours_2)

                        mode = "optimal"
                    if event.key == pygame.K_p:
                        self.factories_act = self.factories
                        self.per_min_act = copy.deepcopy(self.per_min)
                        self.edge_colours_act = copy.deepcopy(self.edge_colours)
                        mode = "normal"
            if dragging:
                pos = pygame.mouse.get_pos()
                if first_drag:
                    pos_last = pos
                diff = (pos[0] - pos_last[0], pos[1] - pos_last[1])
                OFFSET_X += diff[0]
                OFFSET_Y += diff[1]
                first_drag = False
                pos_last = pos
            for edge in self.edges[1:]:
                va, vb = edge
                if self.per_min_act[va] > 0 and self.per_min_act[vb] > 0:
                    color = self.edge_colours_act[edge]
                    pygame.draw.line(screen, color,
                                     (M * positions[va][0] + OFFSET_X, M * positions[va][1] + OFFSET_Y),
                                     (M * positions[vb][0] + OFFSET_X, M * positions[vb][1] + OFFSET_Y))
            if mode == "optimal":
                if self.new_edges:
                    for edge_i, edge in enumerate(self.new_edges):
                        va, vb = edge
                        name_clean = "_".join(self.new_edges_back[edge_i][1].split("_")[:-1])
                        color = self.edge_colours_excess[edge_i]

                        pygame.draw.line(screen, color,
                                         (M * positions[va][0] + OFFSET_X + M * 20 * (
                                                 (positions[vb][1] - positions[va][1]) / np.sqrt(
                                             (positions[vb][1] - positions[va][1]) ** 2 + (
                                                     positions[vb][0] - positions[va][0]) ** 2)),
                                          M * positions[va][1] + OFFSET_Y - M * 20 * (
                                                  (positions[vb][0] - positions[va][0]) / np.sqrt(
                                              (positions[vb][1] - positions[va][1]) ** 2 + (
                                                      positions[vb][0] - positions[va][0]) ** 2))),
                                         (M * positions[vb][0] + OFFSET_X + M * 20 * (
                                                 (positions[vb][1] - positions[va][1]) / np.sqrt(
                                             (positions[vb][1] - positions[va][1]) ** 2 + (
                                                     positions[vb][0] - positions[va][0]) ** 2)),
                                          M * positions[vb][1] + OFFSET_Y - M * 20 * (
                                                  (positions[vb][0] - positions[va][0]) / np.sqrt(
                                              (positions[vb][1] - positions[va][1]) ** 2 + (
                                                      positions[vb][0] - positions[va][0]) ** 2))))

                        imp = pygame.image.load(Graph.ITEMS[name_clean].img_name).convert_alpha()
                        imp = pygame.transform.scale(imp, (M * 2 * imp.get_width(), M * 2 * imp.get_height()))
                        imp_rect = imp.get_rect()
                        imp_rect.center = ((M * (positions[va][0] + positions[vb][0]) + 2 * OFFSET_X) / 2,
                                           (M * (positions[va][1] + positions[vb][1]) + 2 * OFFSET_Y) / 2)
                        screen.blit(imp, imp_rect)
                        font = pygame.font.Font('freesansbold.ttf', 20)
                        text = font.render(str(self.per_min_excs[edge_i]), True, WHITE)
                        text = pygame.transform.scale(text, (M * 2 * text.get_width(), M * 2 * text.get_height()))

                        textRect = text.get_rect()
                        pygame.transform.scale(text, (M * text.get_width(), M * text.get_height()))
                        # set the center of the rectangular object.
                        textRect.midtop = ((M * (positions[va][0] + positions[vb][0]) + 2 * OFFSET_X) / 2,
                                           (M * (positions[va][1] + positions[vb][
                                               1]) + 2 * OFFSET_Y) / 2 + imp.get_height() // 2)

                        screen.blit(text, textRect)
            for p, key in enumerate(positions):
                name_array = key.split("_")[:-1]
                name_clean = "_".join(name_array)
                if not Graph.ITEMS[name_clean].img_name:
                    if p == 0:
                        pygame.draw.circle(screen, GREEN,
                                           (M * positions[key][0] + OFFSET_X, positions[key][1] + OFFSET_Y),
                                           10)
                    else:
                        pygame.draw.circle(screen, BLUE,
                                           (M * positions[key][0] + OFFSET_X, positions[key][1] + OFFSET_Y),
                                           10)

                    pygame.display.set_caption('image')
                else:
                    if self.per_min_act[key] > 0:
                        imp = pygame.image.load(Graph.ITEMS[name_clean].img_name).convert_alpha()
                        imp = pygame.transform.scale(imp, (M * 2 * imp.get_width(), M * 2 * imp.get_height()))
                        imp_rect = imp.get_rect()
                        imp_rect.center = ((M * positions[key][0] + OFFSET_X),
                                           (M * positions[key][1] + OFFSET_Y))
                        screen.blit(imp, imp_rect)
                        font = pygame.font.Font('freesansbold.ttf', 20)
                        text = font.render(str(self.per_min_act[key]), True, WHITE)
                        text = pygame.transform.scale(text, (M * 2 * text.get_width(), M * 2 * text.get_height()))

                        textRect = text.get_rect()
                        pygame.transform.scale(text, (M * text.get_width(), M * text.get_height()))
                        # set the center of the rectangular object.
                        textRect.midtop = ((M * positions[key][0] + OFFSET_X),
                                           (M * positions[key][1] + OFFSET_Y + imp.get_height() // 2))

                        screen.blit(text, textRect)
                        height = 0
                        if self.factories_act[key]:
                            for i, fact in enumerate(self.factories_act[key][1]):
                                key_clean = "_".join(key.split("_")[:-1])
                                if key_clean == "fire_ice":
                                    pass
                                if fact != 0 and Graph.ITEMS[key_clean].made_in is not None:
                                    text = font.render(str(fact), True, WHITE)
                                    text = pygame.transform.scale(text,
                                                                  (M * 2 * text.get_width(), M * 2 * text.get_height()))

                                    textRect = text.get_rect()
                                    pygame.transform.scale(text, (M * text.get_width(), M * text.get_height()))
                                    # set the center of the rectangular object.
                                    textRect.midbottom = ((M * positions[key][0] + OFFSET_X + text.get_width() // 1.5),
                                                          (M * positions[key][
                                                              1] + OFFSET_Y - imp.get_height() // 2 - height * text.get_height()))

                                    screen.blit(text, textRect)
                                    imp_fact = pygame.image.load(
                                        Graph.factory_images[self.factories_act[key][0]][i]).convert_alpha()
                                    imp_fact = pygame.transform.scale(imp_fact,
                                                                      (text.get_height(), text.get_height()))
                                    imp_fact_rect = imp_fact.get_rect()
                                    imp_fact_rect.midbottom = (
                                        (M * positions[key][0] + OFFSET_X - imp_fact.get_width() // 1.5),
                                        (M * positions[key][
                                            1] + OFFSET_Y - imp.get_height() // 2 - height * text.get_height()))
                                    screen.blit(imp_fact, imp_fact_rect)

                                    height += 1

            pygame.display.flip()
        pygame.quit()
