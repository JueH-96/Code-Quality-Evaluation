class Solution:
    def minimumChairs(self, s: str) -> int:
        people_in_room = 0
        max_people = 0
        for event in s:
            if event == 'E':
                people_in_room += 1
            elif event == 'L':
                people_in_room -= 1
            max_people = max(max_people, people_in_room)
        return max_people