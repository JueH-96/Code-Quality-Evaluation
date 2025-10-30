class Solution:
    def minimumChairs(self, s: str) -> int:
        max_chairs = 0
        current_chairs = 0
        for event in s:
            if event == 'E':
                current_chairs += 1
            elif event == 'L':
                current_chairs -= 1
            max_chairs = max(max_chairs, current_chairs)
        return max_chairs