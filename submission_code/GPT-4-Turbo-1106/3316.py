from typing import List
from itertools import combinations

class Solution:
    def sumOfPowers(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        nums.sort()
        sum_of_powers = 0
        
        # Calculate the sum of minimum absolute differences for each combination
        for combo in combinations(nums, k):
            sum_of_powers += combo[-1] - combo[0]
            sum_of_powers %= MOD
        
        return sum_of_powers