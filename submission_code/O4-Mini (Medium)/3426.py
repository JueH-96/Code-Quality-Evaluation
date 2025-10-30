class Solution:
    def minimumChairs(self, s: str) -> int:
        current = 0
        max_chairs = 0
        for ch in s:
            if ch == 'E':
                current += 1
                max_chairs = max(max_chairs, current)
            else:  # ch == 'L'
                current -= 1
        return max_chairs