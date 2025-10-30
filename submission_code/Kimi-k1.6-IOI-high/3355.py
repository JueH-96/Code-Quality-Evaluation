class Solution:
    def minimumLevels(self, possible: List[int]) -> int:
        n = len(possible)
        prefix = [0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + (2 * possible[i - 1] - 1)
        total = prefix[-1]
        for k in range(1, n):
            if prefix[k] > (total - prefix[k]):
                return k
        return -1