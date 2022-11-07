import pygame
import math

BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
ORANGE = (255, 165, 0)
FPS = 60
size = 1920, 1080
OFFSET_X = size[0] // 2
OFFSET_Y = size[1] // 2
GREEN = (0, 255, 0)


def normal_round(num, n_digits=0):
    if n_digits == 0:
        return int(num + 0.5)
    else:
        digit_value = 10 ** n_digits
        return int(num * digit_value + 0.5) / digit_value


class Game:
    MINING_SPEED = 1


class Building():
    building_count = 0

    def __init__(self, size, idle_consumption, work_consumption):
        self.size = size
        self.idle_consumption = idle_consumption
        self.work_consumption = work_consumption

        Building.add_building()

    @classmethod
    def add_building(cls):
        Building.building_count += 1


class MiningMachine(Building):

    def __init__(self, veins):
        super().__init__((3, 3), 24, 420)
        self.gathering_speed = 30 * veins * Game.MINING_SPEED


class WaterPump(Building):

    def __init__(self):
        super().__init__((3, 3), 12, 300)
        self.gathering_speed = 50 * Game.MINING_SPEED


class OilExtractor(Building):

    def __init__(self, oil_field_speed):
        super().__init__((3, 3), 24, 840)
        self.gathering_speed = normal_round(oil_field_speed * Game.MINING_SPEED * 60, 1)


class AdvancedMiningMachine(Building):

    def __init__(self, veins):
        super().__init__((3, 3), 168, 2_940)
        self.gathering_speed = 30 * veins * Game.MINING_SPEED


class ArcSmelter(Building):
    def __init__(self):
        super().__init__((3, 3), 12, 360)
        self.production_speed = 1


class PlaneSmelter(Building):
    def __init__(self):
        super().__init__((3, 3), 48, 1_440)
        self.production_speed = 2


class AssemblingMachineMK1(Building):
    def __init__(self):
        super().__init__((4, 4), 12, 270)
        self.production_speed = 0.75


class AssemblingMachineMK2(Building):
    def __init__(self):
        super().__init__((4, 4), 15, 540)
        self.production_speed = 1


class AssemblingMachineMK3(Building):
    def __init__(self):
        super().__init__((4, 4), 18, 1_080)
        self.production_speed = 1.5


class OilRefinery(Building):
    def __init__(self):
        super().__init__((3, 3), 24, 960)
        self.production_speed = 1


class ChemicalPlant(Building):
    def __init__(self):
        super().__init__((3, 3), 24, 720)
        self.production_speed = 1


class Item:
    def __init__(self, name, n, components=None, n_com=None):
        self.name = name
        self.components = components


class Graph:
    ITEMS = {
        "gear": Item("gear", 1, ["iron_ingot"], [1]),
        "iron_ingot": Item("iron_ingot", 1, ["iron_ore"], [1]),
        "iron_ore": Item("iron_ore", 1, None, None),
        "copper_ore": Item("copper_ore", 1, None, None),
        "magnet": Item("magnet", 1, ["iron_ore"], [1]),
        "copper_ingot": Item("copper_ingot", 1, ["copper_ore"], [1]),
        "magnetic_coil": Item("magnetic_coil", 2, ["magnet", "copper_ingot"], [2, 1]),
        "electric_motor": Item("electric_motor", 1, ["iron_ingot", "gear", "magnetic_coil"], [2, 1, 1])
    }

    def __init__(self, objective):
        self.objective = objective

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
                prev_path = Graph(comp).find_edges(name_ind, prev_path, counts)
        return prev_path

    def get_paths(self):
        path = {}
        self.edges = self.find_edges()
        for edge in self.edges:
            if edge[0]:
                if edge[0] not in path.keys():q
                    path[str(edge[0])] = [edge[1]]
                else:
                    path[str(edge[0])].append(edge[1])
        return path

    def draw(self):
        global OFFSET_X
        global OFFSET_Y
        paths = self.get_paths()
        print(paths)
        print(len(paths))
        paths_num = {}
        ANGLE = 45
        LEN = 100
        positions = {}
        for p, key in enumerate(paths.keys()):
            lines = len(paths[key])
            if lines % 2 == 0:
                angles = [ANGLE // (i + 1) for i in range(lines // 2)]
                angles += [-an for an in angles]

            else:
                angles = [ANGLE // (i + 1) for i in range((lines) // 2)]
                angles += [-an for an in angles]
                angles += [0]
            angles = sorted(angles)
            if p == 0:
                positions[key] = (0, 0)
            for i, comp in enumerate(paths[key]):
                if comp not in positions:
                    positions[comp] = (positions[key][0] - math.cos(math.radians(angles[i])) * LEN,
                                       positions[key][1] - math.sin(math.radians(angles[i])) * LEN)
            LEN -= 10
        print(positions)
        print(len(positions))
        pygame.init()
        screen = pygame.display.set_mode(size)
        pygame.display.flip()
        run = True
        dragging = False
        clock = pygame.time.Clock()
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

            if dragging:
                pos = pygame.mouse.get_pos()
                if first_drag:
                    pos_last = pos
                diff = (pos[0] - pos_last[0], pos[1] - pos_last[1])
                OFFSET_X += diff[0]
                OFFSET_Y += diff[1]
                first_drag = False
                pos_last = pos
            for p, key in enumerate(positions):
                try:
                    for comp in paths[key]:
                        pygame.draw.line(screen, ORANGE, (positions[key][0] + OFFSET_X, positions[key][1] + OFFSET_Y),
                                         (positions[comp][0] + OFFSET_X, positions[comp][1] + OFFSET_Y))
                except:
                    pass
                if p == 0:
                    pygame.draw.circle(screen, GREEN, (positions[key][0] + OFFSET_X, positions[key][1] + OFFSET_Y), 10)
                else:
                    print(key,positions[key])
                    pygame.draw.circle(screen, BLUE, (positions[key][0] + OFFSET_X, positions[key][1] + OFFSET_Y), 10)

            pygame.display.flip()
        pygame.quit()


line = Graph("electric_motor")
line.draw()
