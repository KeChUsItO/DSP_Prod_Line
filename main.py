import pygame
import copy
import itertools
import math
import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from tree_node import Tree_node
from gym import spaces
import os
import warnings
import pickle

import cloudpickle
import multiprocessing as mp

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
mp.connection.REDUCTION = cloudpickle

warnings.filterwarnings('ignore')
os.environ["TF_DEVICE_NAME"] = "/device:GPU:0"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
FPS = 60
size = 1920 * 0.75, 1080 * 0.75 - 31
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


def redondear_n_decimal(numero, n):
    return math.ceil(numero * 10 ** n) / 10 ** n


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
        if excesses:
            self.excesses = excesses
        else:
            if self.n_com:
                self.excesses = [[[], []]] * len(self.n_com)
            else:
                self.excesses = [[[], []]]

        if img_name:
            self.img_name = "Icons/" + img_name
        else:
            self.img_name = None


selection = "silicon_ore"


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
        if min(info["observation"]) < -10 / np.e:
            reward = min(info["observation"])
        elif 0 > min(info["observation"]) > -10 / np.e:
            reward = -10 / np.e
        else:
            reward = 10 * np.e ** -(count / self.count_original) - 10 / np.e
        info["tree"] = copy.deepcopy(self.tree)
        info["sends2"] = copy.deepcopy(sends2)
        info["sends"] = copy.deepcopy(sends)
        return obs, reward, done, info

    def render(self, mode='human'):
        self.tree.draw()

    def return_tree(self):
        return self.tree


class Graph:
    MAX_ASSEMBLER_TIER = 1
    MAX_SMELTING_TIER = 1
    MAX_MINING_TIER = 1
    MAX_CHEMICAL_TIER = 1
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

        "conveyor_belt_mk.I": Item("conveyor_belt_mk.I", 3, 1, ["iron_ingot", "gear"], [2, 1],
                                   "Icon_Conveyor_Belt_Mk.I.png",
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

    def recalc(self):
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
        print("ALL PATHS", get_all_paths(root)[ind])
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
                print(item,Graph.ITEMS[item].excesses, n_com_ind)
                for excess_name_i, excess_name in enumerate(Graph.ITEMS[item].excesses[n_com_ind][0]) :
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
        bestreward = -np.inf

        def make_env():
            return Excess_manager(n_acts, n_obs, fact_tree, original_tree, dict_potential, excesses, dict_excesses,
                                  relation_dict, self.per_min)

        if n_obs > 0:
            env = DummyVecEnv([make_env for _ in range(1)])
            model = PPO("MlpPolicy", env, verbose=1)
            obs = env.reset()
            for i in range(5):
                action, _states = model.predict(obs)
                print(action)
                obs, rewards, done, info = env.step(action)

                if np.any(done):
                    break
            print(obs, rewards, done, action, info)
            env.close()
            rewstd = -1
            rewmean = -np.inf
            min_delta = 0.001
            patience = 30
            prev_mean_reward = -np.inf
            no_improvement_counter = 0
            while rewstd > 10 ** -6 or np.round(np.mean(rewmean), 1) < 0:
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
                rewmean_avg = np.mean(rewmean)
                rewstd = np.std(rewmean)
                if max(rewmean) > bestreward:
                    bestreward = max(rewmean)
                    besttree = copy.deepcopy(trees[np.argmax(np.array(rewmean))])
                    best_send2 = copy.deepcopy(sends2[np.argmax(np.array(rewmean))])
                    best_send = copy.deepcopy(sends[np.argmax(np.array(rewmean))])
                    best_action = copy.deepcopy(actions[np.argmax(np.array(rewmean))])
                print("MEAN", np.round(np.mean(rewmean), 1), "+/-", rewstd,
                      np.round(rewstd / np.mean(rewmean), 2))

                if abs(rewmean_avg - prev_mean_reward) < min_delta and rewmean_avg >= 0:
                    no_improvement_counter += 1
                else:
                    no_improvement_counter = 0

                if no_improvement_counter >= patience:
                    break
                prev_mean_reward = rewmean_avg
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
                        text = font.render(str(redondear_n_decimal(self.per_min_excs[edge_i], 2)), True, WHITE)
                        text = pygame.transform.scale(text, (M * 2 * text.get_width(), M * 2 * text.get_height()))

                        textRect = text.get_rect()
                        pygame.transform.scale(text, (M * text.get_width(), M * text.get_height()))
                        # set the center of the rectangular object.
                        textRect.midtop = ((M * (positions[va][0] + positions[vb][0]) + 2 * OFFSET_X) / 2,
                                           (M * (positions[va][1] + positions[vb][
                                               1]) + 2 * OFFSET_Y) / 2 + imp.get_height() // 2)

                        screen.blit(text, textRect)
            prime_material = {}
            total_fact_dict = {}
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
                        if not Graph.ITEMS[name_clean].components:
                            if name_clean in prime_material.keys():
                                prime_material[name_clean] += self.per_min_act[key]
                            else:
                                prime_material[name_clean] = self.per_min_act[key]

                        text = font.render(str(redondear_n_decimal(self.per_min_act[key], 2)), True, WHITE)
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
                                    if fact > 0:
                                        if self.factories_act[key][0] + "_" + str(i) in total_fact_dict.keys():
                                            total_fact_dict[self.factories_act[key][0] + "_" + str(i)] += fact
                                        else:
                                            total_fact_dict[self.factories_act[key][0] + "_" + str(i)] = fact

                                    imp_fact = pygame.transform.scale(imp_fact,
                                                                      (text.get_height(), text.get_height()))
                                    imp_fact_rect = imp_fact.get_rect()
                                    imp_fact_rect.midbottom = (
                                        (M * positions[key][0] + OFFSET_X - imp_fact.get_width() // 1.5),
                                        (M * positions[key][
                                            1] + OFFSET_Y - imp.get_height() // 2 - height * text.get_height()))
                                    screen.blit(imp_fact, imp_fact_rect)

                                    height += 1
            for key_i, key in enumerate(prime_material.keys()):
                if prime_material[key] > 0:
                    imp = pygame.image.load(Graph.ITEMS[key].img_name).convert_alpha()
                    imp = pygame.transform.scale(imp, (0.5 * imp.get_width(), 0.5 * imp.get_height()))
                    imp_rect = imp.get_rect()
                    imp_rect.center = (0.75 * imp.get_width(),
                                       (key_i * 1.5 + 0.5) * imp.get_height())
                    screen.blit(imp, imp_rect)

                    font = pygame.font.Font('freesansbold.ttf', 12)
                    text = font.render(str(redondear_n_decimal(prime_material[key], 2)), True, WHITE)
                    text = pygame.transform.scale(text, (text.get_width(), text.get_height()))

                    textRect = text.get_rect()
                    pygame.transform.scale(text, (0.5 * text.get_width(), 0.5 * text.get_height()))
                    # set the center of the rectangular object.
                    textRect.midleft = (1.4 * imp.get_width(),
                                        (key_i * 1.5 + 0.5) * imp.get_height())

                    screen.blit(text, textRect)
            last_height = 0
            for key_i, key in enumerate(total_fact_dict.keys()):
                fact_name = "_".join(key.split("_")[:-1])
                fact_id = key.split("_")[-1]
                imp = pygame.image.load(Graph.factory_images[fact_name][int(fact_id)]).convert_alpha()
                imp = pygame.transform.scale(imp, (0.5 * imp.get_width(), 0.5 * imp.get_height()))
                imp_rect = imp.get_rect()
                imp_rect.center = (screen.get_width() - 0.75 * imp.get_width(),
                                   last_height + imp.get_height() / 2)
                screen.blit(imp, imp_rect)

                font = pygame.font.Font('freesansbold.ttf', 12)
                text = font.render(str(total_fact_dict[key]), True, WHITE)
                text = pygame.transform.scale(text, (text.get_width(), text.get_height()))

                textRect = text.get_rect()
                pygame.transform.scale(text, (0.5 * text.get_width(), 0.5 * text.get_height()))
                # set the center of the rectangular object.
                textRect.midright = (screen.get_width() - 1.4 * imp.get_width(),
                                     last_height + imp.get_height() / 2)

                screen.blit(text, textRect)
                last_height += imp.get_height()
            pygame.display.flip()
        pygame.quit()


def build_fact(fact):
    import pygame
    import pygame.locals

    to_place = []
    place_id = 0
    in_place_id = 0
    can_place_fact = True
    for fact_nums in fact.factories2:
        if Graph.ITEMS["_".join(fact_nums.split('_')[:-1])].n_com:
            print(fact.factories2)
            if fact.factories2[fact_nums][0] not in ['mining_machine', 'oil_extraction_facility',
                                                     'water_pumping_facility']:
                to_place.append(fact.factories2[fact_nums])
    print("To place:", to_place)
    print("Edge Colours", fact.edge_colours_2)
    to_place_next = to_place[place_id][1][in_place_id]
    to_place_next_kind = list(Graph.factory_images.keys()).index(
        to_place[place_id][0]) + (in_place_id + 1) / 10
    # Definir dimensiones del grid
    grid_width = 100
    grid_height = 100

    # Definir tamaño de cada celda
    cell_size = 10
    border_size = 1  # Bordas más finas

    # Crear dos grids 2D llenos de ceros
    grid1 = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    grid2 = [[0 for _ in range(grid_width + 1)] for _ in range(grid_height + 1)]  # Grid2 es más grande
    grid_imgs = [[0 for _ in range(grid_width + 1)] for _ in range(grid_height + 1)]

    # Variables de desplazamiento y zoom
    scroll = [0, 0]
    zoom_level = 1

    # Función para cambiar el valor de una celda

    # Inicializar Pygame
    pygame.init()

    # Crear ventana de Pygame
    window = pygame.display.set_mode((800, 600))

    # Colores
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    # Variables para el control de scroll del ratón
    dragging = False
    last_mouse_pos = (0, 0)

    # Crear una superficie transparente para el segundo grid
    grid2_surface = pygame.Surface((800, 600))  # Ajustar tamaño de la superficie
    grid2_surface.set_colorkey(BLACK)

    # Bucle de juego
    running = True

    def check_cells(grid, x, y):
        # Verificar si todas las celdas en el rango 3x3 pueden ser llenadas
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                new_x = x + dx + 1
                new_y = y + dy + 1
                # Si alguna celda está fuera del grid o ya tiene un valor, no permita la colocación
                if not (0 <= new_x < len(grid[0]) and 0 <= new_y < len(grid)) or grid[new_y][new_x] != 0:
                    return False
        return True

    def set_cell(grid, x, y, value):
        if 0 <= x < len(grid[0]) and 0 <= y < len(grid):  # Ajustar para el tamaño del grid
            if grid[y][x] == 0:  # Solo cambia el valor si la celda actualmente tiene un valor de 0
                grid[y][x] = value

    running = True
    while running:
        # Manejar eventos de Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Botón izquierdo
                    dragging = True
                    last_mouse_pos = event.pos
                    # Convertir posición del ratón a coordenadas del grid
                    cell_x = round((event.pos[0] - scroll[0] - cell_size * zoom_level / 2) // (cell_size * zoom_level))
                    cell_y = round((event.pos[1] - scroll[1] - cell_size * zoom_level / 2) // (cell_size * zoom_level))
                    # Establecer valores en el grid
                    if can_place_fact:
                        # Solo intenta establecer el valor si las coordenadas están dentro del rango del grid
                        if check_cells(grid2, cell_x, cell_y):
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    set_cell(grid2, cell_x + dx + 1, cell_y + dy + 1,
                                             to_place_next_kind)  # Actualizar el grid2

                            if 0 <= cell_x < len(grid_imgs[0]) and 0 <= cell_y < len(grid_imgs):
                                grid_imgs[cell_y][cell_x] = Graph.factory_images[to_place[place_id][0]][in_place_id]
                                print(grid_imgs[cell_y][cell_y])
                                print(to_place_next)
                                to_place_next -= 1
                                if to_place_next == 0:
                                    while to_place_next == 0:
                                        in_place_id += 1
                                        try:
                                            to_place_next = to_place[place_id][1][in_place_id]
                                            to_place_next_kind = list(Graph.factory_images.keys()).index(
                                                to_place[place_id][0]) + (in_place_id + 1) / 10
                                        except:
                                            in_place_id = 0
                                            place_id += 1
                                            if place_id < len(to_place):
                                                to_place_next = to_place[place_id][1][in_place_id]
                                                to_place_next_kind = list(Graph.factory_images.keys()).index(
                                                    to_place[place_id][0]) + (in_place_id + 1) / 10
                                            else:
                                                can_place_fact = False
                                                np.set_printoptions(threshold=np.inf)
                                                to_place_next = -1
                                    print("To place next:", to_place)



                elif event.button == 4:  # Rueda del ratón hacia arriba
                    zoom_level += 0.1

                elif event.button == 5:  # Rueda del ratón hacia abajo
                    zoom_level = max(0.1, zoom_level - 0.1)  # Evitar el zoom level negativo o cero

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:  # Botón izquierdo
                    dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging:
                    dx = event.pos[0] - last_mouse_pos[0]
                    dy = event.pos[1] - last_mouse_pos[1]
                    scroll[0] += dx
                    scroll[1] += dy
                    last_mouse_pos = event.pos

        # Dibujar los grids
        window.fill(BLACK)
        cell_draw_size = cell_size * zoom_level
        border_draw_size = border_size * zoom_level
        image_cache = {}  # Cache de todas las imágenes
        for y in range(grid_height + 1):
            for x in range(grid_width + 1):
                image_name = grid_imgs[y][x]
                if image_name and image_name not in image_cache:
                    image_cache[image_name] = pygame.image.load(image_name)

        # Dibujar el primer grid (grid de fondo)
        for y in range(grid_height):
            for x in range(grid_width):
                rect = pygame.Rect(
                    x * cell_draw_size + scroll[0],
                    y * cell_draw_size + scroll[1],
                    cell_draw_size,
                    cell_draw_size)
                pygame.draw.rect(window, WHITE, rect)  # Bordas blancas
                inner_rect = pygame.Rect(
                    rect.left + border_draw_size,
                    rect.top + border_draw_size,
                    cell_draw_size - 2 * border_draw_size,
                    cell_draw_size - 2 * border_draw_size)
                pygame.draw.rect(window, WHITE if grid1[y][x] else BLACK, inner_rect)

        # Dibujar el segundo grid (desplazado medio cuadro)
        grid2_surface.fill(BLACK)  # Limpiar la superficie antes de dibujar
        for y in range(grid_height + 1):  # Ajustar para el tamaño del grid
            for x in range(grid_width + 1):  # Ajustar para el tamaño del grid
                rect = pygame.Rect(
                    x * cell_draw_size + scroll[0] - cell_draw_size / 2,
                    # Ajustar para el desplazamiento medio cuadro en x
                    y * cell_draw_size + scroll[1] - cell_draw_size / 2,
                    # Ajustar para el desplazamiento medio cuadro en y
                    cell_draw_size,
                    cell_draw_size)
                # pygame.draw.rect(grid2_surface, WHITE, rect)  # Bordas blancas
                inner_rect = pygame.Rect(
                    rect.left,
                    rect.top,
                    cell_draw_size,
                    cell_draw_size)
                color = {
                    0.1: ORANGE,
                    0.2: GR_BL,
                    0.3: BLUE
                }.get(normal_round(grid2[y][x] % 1, 2), BLACK)
                pygame.draw.rect(grid2_surface, color, inner_rect)
        # Dibujar la superficie del segundo grid en la ventana
        window.blit(grid2_surface, (0, 0))  # Mismas coordenadas que para dibujar grid1
        for y in range(grid_height + 1):
            for x in range(grid_width + 1):
                if grid_imgs[y][x]:  # Si hay una imagen para esta celda
                    # Obtén la imagen de la caché
                    image = image_cache[grid_imgs[y][x]]
                    # Ajusta el tamaño de la imagen al tamaño de la celda
                    image = pygame.transform.scale(image, (int(cell_draw_size), int(cell_draw_size)))
                    # Crea un rectángulo para la imagen
                    image_rect = image.get_rect()
                    # Define la posición del rectángulo (imagen)
                    image_rect.center = ((x + 1.5) * cell_draw_size + scroll[0] - cell_draw_size / 2,
                                         (y + 1.5) * cell_draw_size + scroll[1] - cell_draw_size / 2)
                    # Dibuja la imagen en la ventana
                    window.blit(image, image_rect)

        pygame.display.update()

    # Cerrar Pygame
    pygame.quit()


if __name__ == '__main__':
    DONT_HAVE = ['sulfuric_acid_ocean', 'fire_ice_vein']
    selection = "arc_smelter"
    line = Graph(selection, 120)
    line.calculate(ind=0)

    filename = f"Factories/{selection}.pickle"

    # Verificar si el archivo existe y se puede escribir en él
    if not os.path.exists(filename):
        try:
            with open(filename, "wb") as archivo:
                archivo.close()
        except Exception as e:
            print(f"Error al crear archivo: {e}")
            exit()

    contador = 0
    print(filename)
    idxs = []
    with open(filename, 'rb') as archivo:
        fact_nums = {}
        objs = {}

        while True:
            not_add = False
            try:
                objeto = pickle.load(archivo)
                objs[objeto.ind] = objeto
                idxs.append(objeto.ind)
                obj_vertexes = []
                for vertex in objeto.vertexes:
                    obj_vertexes.append("_".join(vertex.split('_')[:-1]))
                for dont in DONT_HAVE:
                    if dont in obj_vertexes:
                        not_add = True
                if not_add:
                    continue
                val1 = objeto.count_factories(objeto.factories)
                try:
                    val2 = objeto.count_factories(objeto.factories2)
                except:
                    val2 = val1
                fact_nums[objeto.ind] = min(val1, val2)
                contador += 1
            except EOFError:
                break
        archivo.close()
        sorted_fact_nums = dict(sorted(fact_nums.items(), key=lambda item: item[1]))

    for key in sorted_fact_nums.keys():
        objs[key].recalc()
        objs[key].draw()
        # build_fact(objs[key])
    print(contador)
    with open(filename, 'wb') as archivo:  # Asegúrate de abrir el archivo en modo 'wb' para escribir en binario
        for ind_count in range(contador, line.all_posibilities):
            line.calculate(ind=ind_count)
            pickle.dump(line, archivo)  # Guardar el objeto line en el archivo
            ind_count += 1
