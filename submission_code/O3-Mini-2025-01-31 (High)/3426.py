class Solution:
    def minimumChairs(self, s: str) -> int:
        current = 0
        max_count = 0
        for char in s:
            if char == 'E':
                current += 1
                max_count = max(max_count, current)
            else:  # char == 'L'
                current -= 1
        return max_count