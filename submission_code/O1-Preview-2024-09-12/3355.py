class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        total_sum = sum(possible)
        D = -2 * total_sum + n
        for k in range(1, n):
            D += 4 * possible[k - 1] - 2
            if D > 0:
                return k
        return -1