class Solution:
    def numberOfChild(self, n: int, k: int) -> int:
        child = 0      # start with child 0
        direction = 1  # initial direction is towards the right
        for _ in range(k):
            child += direction
            # if we are at an end, reverse direction
            if child == 0 or child == n - 1:
                direction *= -1
        return child