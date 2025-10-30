class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        total = 0
        
        # Generate all subsequences of length k using indices
        from itertools import combinations
        
        for indices in combinations(range(n), k):
            # Get the values at these indices
            subsequence = [nums[i] for i in indices]
            
            # Find the minimum absolute difference in this subsequence
            min_diff = float('inf')
            for i in range(k):
                for j in range(i + 1, k):
                    min_diff = min(min_diff, abs(subsequence[i] - subsequence[j]))
            
            total = (total + min_diff) % MOD
        
        return total