trainId_to_name = {
    0: "unlabeled",
    1: "ego vehicle",
    2: "rectification border",
    3: "out of roi",
    4: "static",
    5: "dynamic",
    6: "ground",
    7: "road",
    8: "sidewalk",
    9: "parking",
    10: "rail track",
    11: "building",
    12: "wall",
    13: "fence",
    14: "guard rail",
    15: "bridge",
    16: "tunnel",
    17: "pole",
    18: "polegroup",
    19: "traffic light",
    20: "traffic sign",
    21: "vegetation",
    22: "terrain",
    23: "sky",
    24: "person",
    25: "rider",
    26: "car",
    27: "truck",
    28: "bus",
    29: "caravan",
    30: "trailer",
    31: "train",
    32: "motorcycle",
    33: "bicycle",
    -1: "license plate",
}



color_to_trainId = {
    (0, 0, 0): 255,    # unlabeled
    (0, 0, 0): 255,    # ego vehicle
    (0, 0, 0): 255,    # rectification border
    (0, 0, 0): 255,    # out of roi
    (0, 0, 0): 255,    # static
    (111, 74, 0): 255,    # dynamic
    (81, 0, 81): 255,    # ground
    (128, 64, 128): 0,    # road
    (244, 35, 232): 1,    # sidewalk
    (250, 170, 160): 255,   # parking
    (230, 150, 140): 255,   # rail track
    (70, 70, 70): 2,   # building
    (102, 102, 156): 3,   # wall
    (190, 153, 153): 4,   # fence
    (180, 165, 180): 255,   # guard rail
    (150, 100, 100): 255,   # bridge
    (150, 120, 90): 255,   # tunnel
    (153, 153, 153): 5,   # pole
    (153, 153, 153): 255,   # polegroup
    (250, 170, 30): 6,   # traffic light
    (220, 220, 0): 7,   # traffic sign
    (107, 142, 35): 8,   # vegetation
    (152, 251, 152): 9,   # terrain
    (70, 130, 180): 10,   # sky
    (220, 20, 60): 11,   # person
    (255, 0, 0): 12,   # rider
    (0, 0, 142): 13,   # car
    (0, 0, 70): 14,   # truck
    (0, 60, 100): 15,   # bus
    (0, 0, 90): 255,   # caravan
    (0, 0, 110): 255,   # trailer
    (0, 80, 100): 16,   # train
    (0, 0, 230): 17,   # motorcycle
    (119, 11, 32): 18,   # bicycle
    (0, 0, 142): -1,   # license plate
}
