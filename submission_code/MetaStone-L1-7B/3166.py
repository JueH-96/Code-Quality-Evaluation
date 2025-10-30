from collections import Counter

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        freq = Counter(nums)
        max_freq = max(freq.values())
        candidates = []
        for m in range(1, max_freq + 1):
            T = 0
            for k in freq.values():
                T += (k + m - 1) // m  # Equivalent to ceil(k/m)
            x = n - T * m
            if 0 <= x <= T:
                candidates.append(T)
        return min(candidates)