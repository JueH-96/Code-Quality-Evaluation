class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        from itertools import combinations
        MOD = 10**9 + 7
        
        total_power = 0
        n = len(nums)
        
        # Generate all possible non-empty subsequences
        for r in range(1, n + 1):
            for subseq in combinations(nums, r):
                if sum(subseq) == k:
                    total_power = (total_power + 1) % MOD
        
        # Each subsequence contributes its power to the total
        return total_power