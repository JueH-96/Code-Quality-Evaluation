class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        from itertools import combinations
        MOD = 10**9 + 7
        total = 0
        
        # Generate all subsequences of length k
        for subsequence in combinations(nums, k):
            # Sort the subsequence
            sorted_subseq = sorted(subsequence)
            
            # Find the minimum absolute difference between adjacent elements
            min_diff = float('inf')
            for i in range(1, k):
                diff = sorted_subseq[i] - sorted_subseq[i-1]
                min_diff = min(min_diff, diff)
            
            # Add to total
            total = (total + min_diff) % MOD
        
        return total