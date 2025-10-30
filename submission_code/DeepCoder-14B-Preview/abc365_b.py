class Solution:
    def minimumChairs(self, s: str) -> int:
        current = 0
        max_chairs = 0
        for c in s:
            if c == 'E':
                current += 1
            else:
                current -= 1
            if current > max_chairs:
                max_chairs = current
        return max_chairs