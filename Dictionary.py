import json

class Entry:
    def __init__(self, entry, key=None, parent=None, x=0, y=0):
        self.entry = entry
        self.parent = parent
        self.x = x
        self.y = y
        if key is None:
            self.key = self.entry
        else:
            self.key = key
        self.left_child = None
        self.right_child = None
        
class Dictionary:
    def __init__(self):
        self.lines = []
        self.entries = {}
        self.heigth = 0
        self.y_start = 0

def generate_dictionary(fname):
    # with open(fname) as f:
    KEY_LIST = 'AEOSR INDMU TCLPV GHQBF ZJXKW Y'
    CHAR_LIST = 'AEOSR INDMU TCLPV GHQBF ZJXKW Y'
    
    N = 1
    n = 0
    current_line_y = 0.2
    line_x_spacing = 0.1
    line_y_spacing = 0.09
    
    letter = Entry(CHAR_LIST[0], x=0.5, y=0.1)
    new_dictionary = Dictionary()
    new_dictionary.entries[letter.key] = letter
    new_dictionary.lines = [[letter]]
    new_dictionary.y_start = current_line_y
    new_dictionary.heigth = current_line_y + line_y_spacing
    
    new_line = []
    for k, c in zip(KEY_LIST[1:],CHAR_LIST[1:]):
        if c==' ':
            continue
        
        parent = new_dictionary.lines[-1][n]
        if parent.left_child is None:
            x = parent.x - line_x_spacing/N
            letter = Entry(entry=c, parent=parent, x=x, y=current_line_y)
            parent.left_child = letter
        else:
            x = parent.x + line_x_spacing/N
            letter = Entry(entry=c, parent=parent, x=x, y=current_line_y)
            parent.right_child = letter
            n += 1
        
        new_dictionary.entries[k] = letter
        new_line.append(letter)
        
        if n == N:
            n = 0
            N *= 2
            current_line_y += line_y_spacing
            new_dictionary.lines.append(new_line)
            new_dictionary.heigth += line_y_spacing
            new_line = []
    
    parent = new_dictionary.lines[-1][n]
    if parent.left_child is None:
        x = parent.x - line_x_spacing/N
        letter = Entry(entry=' ', key='SPACE', parent=parent, x=x, y=current_line_y)
        parent.left_child = letter
    else:
        x = parent.x + line_x_spacing/N
        letter = Entry(entry=' ', key='SPACE', parent=parent, x=x, y=current_line_y)
        parent.right_child = letter
        n += 1
    new_line.append(letter)
    new_dictionary.lines.append(new_line)
    new_dictionary.heigth += line_y_spacing
    
    return new_dictionary