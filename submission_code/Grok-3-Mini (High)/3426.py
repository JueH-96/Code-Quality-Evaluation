class Solution:
    def minimumChairs(self, s: str) -> int:
        max_chairs = 0
        current_people = 0
        for event in s:
            if event == 'E':
                current_people += 1
                if current_people > max_chairs:
                    max_chairs = current_people
            else:  # event == 'L'
                current_people -= 1
        return max_chairs