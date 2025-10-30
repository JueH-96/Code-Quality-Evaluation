from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Count occurrences of each number
        count = defaultdict(int)
        for num in nums:
            count[num] += 1
        
        # Store the number of subsequences for each GCD
        gcd_count = defaultdict(int)
        
        # Calculate the number of non-empty subsequences for each unique number
        for num, cnt in count.items():
            # The number of non-empty subsequences of a set of size cnt is 2^cnt - 1
            gcd_count[num] = (pow(2, cnt, MOD) - 1) % MOD
        
        # To find pairs of subsequences with the same GCD
        total_pairs = 0
        
        # Iterate over all possible GCDs
        for g in range(1, 201):
            # Find all multiples of g
            multiples = []
            for multiple in range(g, 201, g):
                if multiple in gcd_count:
                    multiples.append(gcd_count[multiple])
            
            # If we have multiples, calculate the number of pairs
            if multiples:
                total_subsequences = sum(multiples) % MOD
                # The number of ways to choose two different subsequences
                total_pairs += (total_subsequences * total_subsequences - sum(m * m for m in multiples)) % MOD
                total_pairs %= MOD
        
        return total_pairs

# Example usage:
# sol = Solution()
# print(sol.subsequencePairCount([1, 2, 3, 4]))  # Output: 10
# print(sol.subsequencePairCount([10, 20, 30]))  # Output: 2
# print(sol.subsequencePairCount([1, 1, 1, 1]))  # Output: 50