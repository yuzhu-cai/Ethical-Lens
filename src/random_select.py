import numpy as np
import random

gender_list = ['male', 'female']
race_list = ['White', 'Black', 'Latino-Hispanic', 'Asian', 'MiddleEastern']
age_list = ['infancy', 'childhood', 'adolescence', 'young adulthood', 'middle age', 'old age']

class RandomSelect:

    def __init__(self):
        
        self.gender_list = self.shuffle(gender_list)
        self.race_list = self.shuffle(race_list)
        self.age_list = self.shuffle(age_list)


    def refresh_age_list(self, age_range):

        tmp_list = []
        for item in self.age_list:
            if item in age_range:
                tmp_list.append(item)
        self.age_list = tmp_list

    def shuffle(self, b):

        random.shuffle(b)
        return b

    def age_forward(self, age_range, img_number):

        self.refresh_age_list(age_range)
        if len(self.age_list) >= img_number:
            result_age = self.age_list[:img_number]
            self.age_list = [item for item in self.age_list if item not in result_age]
            if self.age_list == []:
                self.age_list = age_range.copy()
                random.shuffle(self.age_list)
        else:
            part1 = self.age_list
            remain_part = self.shuffle([item for item in age_range if item not in part1])
            self.age_list = remain_part + self.shuffle(part1.copy())
            part2 = self.age_forward(age_range, img_number - len(part1))
            result_age = part1 + part2

        return result_age
    
    def gender_forward(self, img_number):

        if len(self.gender_list) >= img_number:
            result_gender = self.gender_list[:img_number]
            self.gender_list = [item for item in self.gender_list if item not in result_gender]
            if self.gender_list == []:
                self.gender_list = gender_list.copy()
                random.shuffle(self.gender_list)
        else:
            part1 = self.gender_list
            remain_part = self.shuffle([item for item in gender_list if item not in part1])
            self.gender_list = remain_part + self.shuffle(part1.copy())
            part2 = self.gender_forward(img_number - len(part1))
            result_gender = part1 + part2

        return result_gender
    
    def race_forward(self, img_number):

        if len(self.race_list) >= img_number:
            result_race = self.race_list[:img_number]
            self.race_list = [item for item in self.race_list if item not in result_race]
            if self.race_list == []:
                self.race_list = race_list.copy()
                random.shuffle(self.race_list)
        else:
            part1 = self.race_list
            remain_part = self.shuffle([item for item in race_list if item not in part1])
            self.race_list = remain_part + self.shuffle(part1.copy())
            part2 = self.race_forward(img_number - len(part1))
            result_race = part1 + part2

        return result_race


if __name__ == "__main__":

    ### python src/random_select.py
    random_select = RandomSelect()
    # import pdb; pdb.set_trace()
    print(random_select.race_forward(1))
    print(random_select)
    # import pdb; pdb.set_trace()