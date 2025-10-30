class Solution:
    def minimumChairs(self, s: str) -> int:
        current_people = 0
        max_people = 0
        for event in s:
            if event == 'E':
                current_people += 1
                if current_people > max_people:
                    max_people = current_people
            else:
                current_people -= 1
        return max_people