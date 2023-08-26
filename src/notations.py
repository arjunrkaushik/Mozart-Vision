class Notation:
    def __init__(self, string, fret, pitch):
        self.fret = fret
        self.string = string
        self.pitch = pitch


notation_map = {
    's1': Notation(1, 3, 'G'),
    's2': Notation(1, 0, 'E'),
    's3': Notation(2, 1, 'C'),
    's4': Notation(3, 2, 'A'),
    's5': Notation(4, 3, 'F'),
    's6': Notation(5, 5, 'D'),
    'l1': Notation(2, 6, 'F'),
    'l2': Notation(2, 3, 'D'),
    'l3': Notation(3, 4, 'B'),
    'l4': Notation(4, 5, 'G'),
    'l5': Notation(4, 2, 'E')

}
