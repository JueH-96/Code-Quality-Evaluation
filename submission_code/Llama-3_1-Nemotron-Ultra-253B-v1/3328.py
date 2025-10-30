class Solution:
    def minOperations(self, k: int) -> int:
        if k == 1:
            return 0
        min_steps = float('inf')
        max_x = int(k**0.5) + 1
        for x in range(1, max_x + 1):
            t = (k + x - 1) // x
            steps = x + t - 2
            if steps < min_steps:
                min_steps = steps
        return min_steps