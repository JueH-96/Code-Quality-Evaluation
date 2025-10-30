class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + possible[i]

        min_levels = n + 1
        for i in range(n):
            alice_points = prefix_sum[i + 1]
            bob_points = prefix_sum[n] - prefix_sum[i + 1]
            if alice_points > bob_points:
                min_levels = min(min_levels, i + 1)

        if min_levels == n + 1:
            return -1
        else:
            return min_levels