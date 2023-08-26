from note import Note


class NotationTablature:
    def __init__(self, note: Note, staff_region):
        self.note = note
        self.staff_region = staff_region
        self.note_type = ""

    def set_note_type(self, notetype):
        self.note_type = notetype
