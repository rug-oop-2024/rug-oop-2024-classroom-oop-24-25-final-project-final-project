from typing import Literal

class Feature:

    def __init__(self, name: str, type: Literal['numerical', 'categorical']):
        self.name = name
        self.type = type

    def __str__(self):
        return f"Feature(name={self.name}, type={self.type})"

    def __repr__(self):
        return self.__str__()

age_feature = Feature(name="age", type="numerical")
print(age_feature)


gender_feature = Feature(name="gender", type="categorical")
print(gender_feature)
