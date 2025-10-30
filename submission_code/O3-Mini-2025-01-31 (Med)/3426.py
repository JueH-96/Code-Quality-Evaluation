class Solution:
    def minimumChairs(self, s: str) -> int:
        current = 0
        max_chairs = 0
        for event in s:
            if event == 'E':
                current += 1
            elif event == 'L':
                current -= 1
            max_chairs = max(max_chairs, current)
        return max_chairs