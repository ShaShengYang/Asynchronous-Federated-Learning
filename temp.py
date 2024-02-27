import copy


class Cat:
    def __init__(self):
        self.name = 'tom'


c0 = Cat()
print(c0)
print(copy.deepcopy(c0))
