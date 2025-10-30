from itertools import combinations

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        power_sum = 0
        
        # Calculate the power of all subsequences
        for i in range(1, len(nums) + 1):
            for subseq in combinations(nums, i):
                if sum(subseq) == k:
                    power_sum += 1
        
        return power_sum % MOD