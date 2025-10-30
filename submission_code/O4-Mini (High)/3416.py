from typing import List

class Solution:
    def sumDigitDifferences(self, nums: List[int]) -> int:
        # Convert all numbers to strings (they all have the same length)
        arr = list(map(str, nums))
        n = len(arr)
        if n < 2:
            return 0
        
        k = len(arr[0])                 # number of digit positions
        total_pairs = n * (n - 1) // 2  # total number of unordered pairs
        result = 0
        
        # For each digit position, count how many pairs differ at that position
        for i in range(k):
            freq = [0] * 10
            for s in arr:
                freq[ord(s[i]) - ord('0')] += 1
            
            # Count how many pairs have the same digit at position i
            same_pairs = 0
            for f in freq:
                same_pairs += f * (f - 1) // 2
            
            # The rest of the pairs differ at this position
            result += (total_pairs - same_pairs)
        
        return result