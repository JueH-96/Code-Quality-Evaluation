class Solution:
    def minimumChairs(self, s: str) -> int:
        current_people = 0
        max_chairs = 0
        for c in s:
            if c == 'E':
                current_people += 1
            else:
                current_people -= 1
            if current_people > max_chairs:
                max_chairs = current_people
        return max_chairs