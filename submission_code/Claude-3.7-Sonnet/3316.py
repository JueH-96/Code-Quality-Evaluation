class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        from itertools import combinations
        MOD = 10**9 + 7
        total_power = 0
        
        for subsequence in combinations(nums, k):
            # Sort the subsequence to easily find min difference
            sorted_sub = sorted(subsequence)
            min_diff = float('inf')
            
            # Find minimum difference between adjacent elements
            for i in range(1, len(sorted_sub)):
                diff = sorted_sub[i] - sorted_sub[i-1]
                min_diff = min(min_diff, diff)
                if min_diff == 0:  # Can't get smaller than 0
                    break
                    
            total_power = (total_power + min_diff) % MOD
        
        return total_power