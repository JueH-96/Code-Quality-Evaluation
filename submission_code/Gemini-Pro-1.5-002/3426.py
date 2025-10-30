class Solution:
    def minimumChairs(self, s: str) -> int:
        count = 0
        max_count = 0
        for char in s:
            if char == 'E':
                count += 1
            elif char == 'L':
                count -= 1
            max_count = max(max_count, count)
        return max_count