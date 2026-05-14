class Letter:
    def __init__(self, char, name=None, parent=None, x=0, y=0):
        self.char = char
        self.parent = parent
        self.x = x
        self.y = y

        if name is None:
            self.name = self.char
        else:
            self.name = name
        self.left_child = None
        self.right_child = None
        
class Dictionary:
    def __init__(self):
        pass
