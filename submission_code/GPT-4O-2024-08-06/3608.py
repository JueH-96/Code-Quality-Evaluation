from typing import List
from math import gcd
from collections import defaultdict

class Solution:
    def subsequencePairCount(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        max_num = max(nums)
        
        # Count frequency of each number in nums
        freq = [0] * (max_num + 1)
        for num in nums:
            freq[num] += 1
        
        # Calculate the number of subsequences for each possible GCD
        subseq_count = [0] * (max_num + 1)
        
        # Iterate over each possible GCD from max_num down to 1
        for g in range(max_num, 0, -1):
            # Count how many numbers are multiples of g
            multiples_count = 0
            for multiple in range(g, max_num + 1, g):
                multiples_count += freq[multiple]
            
            # Calculate the number of non-empty subsequences with GCD g
            if multiples_count > 0:
                subseq_count[g] = (pow(2, multiples_count, MOD) - 1) % MOD
            
            # Subtract subsequences that have a greater GCD
            multiple = 2 * g
            while multiple <= max_num:
                subseq_count[g] = (subseq_count[g] - subseq_count[multiple] + MOD) % MOD
                multiple += g
        
        # Calculate the total number of valid pairs
        total_pairs = 0
        for g in range(1, max_num + 1):
            if subseq_count[g] > 0:
                total_pairs = (total_pairs + subseq_count[g] * subseq_count[g]) % MOD
        
        return total_pairs