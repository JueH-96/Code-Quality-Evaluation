from typing import List
from itertools import combinations

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        total_power = 0
        
        # Iterate over all possible subsequence lengths
        for length in range(1, n + 1):
            # Generate all subsequences of the current length
            for subseq in combinations(nums, length):
                # Calculate the number of subsequences with sum equal to k
                if sum(subseq) == k:
                    total_power += 1
                    total_power %= MOD
        
        return total_power

# Example usage:
# sol = Solution()
# print(sol.sumOfPower([1, 2, 3], 3))  # Output: 6
# print(sol.sumOfPower([2, 3, 3], 5))  # Output: 4
# print(sol.sumOfPower([1, 2, 3], 7))  # Output: 0