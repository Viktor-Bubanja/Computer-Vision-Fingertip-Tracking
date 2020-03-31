class MusicPlayer:
    def __init__(self, frame_height):
        print(self)
        self.frame_height = int(frame_height)
        self.notes = self.construct_notes_mapping()
        self.current_note = "A"

    def play_sound(self):
        print("sound")

    def construct_notes_mapping(self):
        step_size = int(((2/3) * self.frame_height) / 4)
        a_notes = {i: "A" for i in range(4 * step_size, self.frame_height)}
        c_notes = {i: "C" for i in range(3 * step_size, 4 * step_size)}
        d_notes = {i: "D" for i in range(2 * step_size, 3 * step_size)}
        e_notes = {i: "E" for i in range(step_size, 2 * step_size)}
        g_notes = {i: "G" for i in range(0, step_size)}
        return {**a_notes, **c_notes, **d_notes, **e_notes, **g_notes}

    def get_current_note(self, finger_position):
        return self.notes[finger_position]

