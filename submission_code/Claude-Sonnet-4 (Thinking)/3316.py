from itertools import combinations

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        total_sum = 0
        
        # Generate all combinations of k elements
        for subsequence in combinations(nums, k):
            # Sort the subsequence to find minimum difference efficiently
            sorted_subseq = sorted(subsequence)
            
            # Find minimum difference between adjacent elements
            min_diff = min(sorted_subseq[i+1] - sorted_subseq[i] 
                          for i in range(len(sorted_subseq) - 1))
            
            total_sum = (total_sum + min_diff) % MOD
        
        return total_sum