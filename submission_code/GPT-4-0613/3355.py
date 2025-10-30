class Solution:
    def minimumLevels(self, possible):
        n = len(possible)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + (2 * possible[i] - 1)
        min_prefix = min(prefix_sum)
        for i in range(n, 0, -1):
            if prefix_sum[i] - min_prefix > 0:
                return i
        return -1