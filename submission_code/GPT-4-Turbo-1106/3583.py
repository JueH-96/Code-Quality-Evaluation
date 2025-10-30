from typing import List
from math import gcd
from collections import Counter

class Solution:
    def gcdValues(self, nums: List[int], queries: List[int]) -> List[int]:
        # Calculate the frequency of each number in nums
        freq = Counter(nums)
        
        # Calculate all possible gcd values and their frequencies
        gcd_freq = Counter()
        unique_nums = list(freq)
        for i in range(len(unique_nums)):
            for j in range(i, len(unique_nums)):
                if i == j:
                    # If both numbers are the same, we only need to consider the combination once
                    gcd_freq[gcd(unique_nums[i], unique_nums[j])] += freq[unique_nums[i]] * (freq[unique_nums[j]] - 1) // 2
                else:
                    # If numbers are different, we consider both combinations
                    gcd_freq[gcd(unique_nums[i], unique_nums[j])] += freq[unique_nums[i]] * freq[unique_nums[j]]
        
        # Sort the gcd values
        gcd_pairs = sorted(gcd_freq.elements())
        
        # Answer the queries
        answer = [gcd_pairs[query] for query in queries]
        return answer

# Example usage:
# sol = Solution()
# print(sol.gcdValues([2,3,4], [0,2,2]))  # Output: [1, 2, 2]
# print(sol.gcdValues([4,4,2,1], [5,3,1,0]))  # Output: [4, 2, 1, 1]
# print(sol.gcdValues([2,2], [0,0]))  # Output: [2, 2]