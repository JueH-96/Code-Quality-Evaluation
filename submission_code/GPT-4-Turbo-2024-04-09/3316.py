class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        from itertools import combinations
        MOD = 10**9 + 7
        
        def min_abs_difference(subseq):
            if len(subseq) < 2:
                return 0
            subseq_sorted = sorted(subseq)
            min_diff = float('inf')
            for i in range(1, len(subseq_sorted)):
                min_diff = min(min_diff, abs(subseq_sorted[i] - subseq_sorted[i-1]))
            return min_diff
        
        all_subsequences = combinations(nums, k)
        sum_of_powers = 0
        for subseq in all_subsequences:
            sum_of_powers += min_abs_difference(subseq)
            sum_of_powers %= MOD
        
        return sum_of_powers