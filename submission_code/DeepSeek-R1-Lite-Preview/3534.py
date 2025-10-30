from typing import List
from itertools import combinations

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        # Convert all numbers to strings and find the maximum length
        str_nums = [str(num) for num in nums]
        max_length = max(len(num) for num in str_nums)
        # Pad all strings with leading zeros to match the maximum length
        padded_nums = [num.zfill(max_length) for num in str_nums]
        n = len(padded_nums)
        count = 0
        # Iterate through all unique pairs (i, j) where i < j
        for i, j in combinations(range(n), 2):
            s = padded_nums[i]
            t = padded_nums[j]
            if s == t:
                count += 1
            else:
                # Find positions where s and t differ
                diff_indices = [idx for idx in range(max_length) if s[idx] != t[idx]]
                if len(diff_indices) == 2:
                    p, q = diff_indices
                    # Check if swapping makes them equal
                    if s[p] == t[q] and s[q] == t[p]:
                        count += 1
        return count