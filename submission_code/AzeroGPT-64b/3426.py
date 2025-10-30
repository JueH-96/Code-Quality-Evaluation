class Solution:
    def minimumChairs(self, s: str) -> int:
        needed_chairs = 0
        max_people = 0
        people = 0
        for event in s:
            if event == 'E':
                people += 1
            else:
                people -= 1
            needed_chairs += 1 if people > max_people else 0
            max_people = max(max_people, people)
        return needed_chairs