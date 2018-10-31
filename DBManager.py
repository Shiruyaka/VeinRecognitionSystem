import numpy as np
import os
from random import randint
import json

UNIVERSITY_DB_PATH = "D:\\VPBase_corrected"
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
WHICH = ["Left", "Right"]
PLACES = ["Wrist", "Palm"]
SERIES = ["Series_1", "Series_2", "Series_3"]


class DBManager:

    def split_images_to_sets(self, hand_place):
        images_path = os.path.join(UNIVERSITY_DB_PATH, hand_place)
        persons = os.listdir(images_path)

        training_sets = []
        test_sets = []
        for per in persons:
            for which in WHICH:
                person_path = os.path.join(images_path, per, which)

                series_no = randint(0, 2)
                image_no = randint(0, 3)
                for s in SERIES:
                    img_path = os.path.join(person_path, s)
                    images = os.listdir(img_path)
                    print(series_no)
                    if s == "Series_" + str(series_no + 1):
                        training_sets += [os.path.join(img_path, images[image_no])]
                        images.remove(images[image_no])

                    test_sets += [os.path.join(img_path, img) for img in images]
        return training_sets, test_sets

    def create_sets(self, hand_place):
        """ hand_place = {Wrist, Palm} """

        assert (hand_place in PLACES)

        training_set, test_set = self.split_images_to_sets(hand_place)
        self.sets_dict = dict([("training_set", training_set), ("test_set", test_set)])

        with open("sets.json", 'w') as sets:
            json.dump([self.sets_dict], sets)

    def __init__(self):
        """CHANGE TO CHOOSE MORE THAN ONE SET"""

        if "sets.json" in os.listdir(PROJECT_PATH):
            with open("sets.json", "r") as file:
                self.sets_dict = json.load(file)[0]
        else:
            self.sets_dict = {}


d = DBManager()
