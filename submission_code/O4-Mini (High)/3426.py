class Solution:
    def minimumChairs(self, s: str) -> int:
        current = 0
        max_chairs = 0
        for c in s:
            if c == 'E':  # person enters
                current += 1
                max_chairs = max(max_chairs, current)
            else:         # person leaves
                current -= 1
        return max_chairs