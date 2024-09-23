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
    (  0,  0,  0): 0,    # unlabeled
    (  0,  0,  0): 1,    # ego vehicle
    (  0,  0,  0): 2,    # rectification border
    (  0,  0,  0): 3,    # out of roi
    (  0,  0,  0): 4,    # static
    (111, 74,  0): 5,    # dynamic
    ( 81,  0, 81): 6,    # ground
    (128, 64,128): 7,    # road
    (244, 35,232): 8,    # sidewalk
    (250,170,160): 9,    # parking
    (230,150,140): 10,   # rail track
    ( 70, 70, 70): 11,   # building
    (102,102,156): 12,   # wall
    (190,153,153): 13,   # fence
    (180,165,180): 14,   # guard rail
    (150,100,100): 15,   # bridge
    (150,120, 90): 16,   # tunnel
    (153,153,153): 17,   # pole
    (153,153,153): 18,   # polegroup
    (250,170, 30): 19,   # traffic light
    (220,220,  0): 20,   # traffic sign
    (107,142, 35): 21,   # vegetation
    (152,251,152): 22,   # terrain
    ( 70,130,180): 23,   # sky
    (220, 20, 60): 24,   # person
    (255,  0,  0): 25,   # rider
    (  0,  0,142): 26,   # car
    (  0,  0, 70): 27,   # truck
    (  0, 60,100): 28,   # bus
    (  0,  0, 90): 29,   # caravan
    (  0,  0,110): 30,   # trailer
    (  0, 80,100): 31,   # train
    (  0,  0,230): 32,   # motorcycle
    (119, 11, 32): 33,   # bicycle
    (  0,  0,142): -1,   # license plate
}
