from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        n = len(nums)
        
        # Convert every number once to its string representation
        str_nums = list(map(str, nums))
        k = len(str_nums[0])                     # all numbers have the same length
        
        total_pairs = n * (n - 1) // 2           # number of unordered pairs
        ans = 0
        
        # Process every digit position independently
        for pos in range(k):
            freq = [0] * 10                      # frequency of each digit (0-9) at this position
            for s in str_nums:
                freq[ord(s[pos]) - 48] += 1      # ord('0') == 48
            
            # pairs that have equal digits in this position
            same_pairs = sum(f * (f - 1) // 2 for f in freq)
            
            # pairs that have different digits in this position
            ans += total_pairs - same_pairs
        
        return ans