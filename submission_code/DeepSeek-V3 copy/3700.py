from collections import defaultdict
from math import comb

class Solution:
    def subsequencesWithMiddleMode(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n < 5:
            return 0
        
        # Precompute the frequency of each number
        freq = defaultdict(int)
        for num in nums:
            freq[num] += 1
        
        result = 0
        
        # Iterate through each possible middle element
        for mid in freq:
            # Count the number of ways to choose 2 elements before mid and 2 after mid
            # such that the frequency of mid is greater than any other element in the subsequence
            # and mid is the unique mode
            
            # First, count the number of elements less than mid and greater than mid
            less = 0
            greater = 0
            for num in freq:
                if num < mid:
                    less += freq[num]
                elif num > mid:
                    greater += freq[num]
            
            # We need to choose 2 elements from the less and 2 from the greater
            # and the middle element is mid
            # The total number of ways is C(less, 2) * C(greater, 2) * freq[mid]
            # But we need to ensure that mid is the unique mode in the subsequence
            # So, the frequency of mid in the subsequence must be 1 (since it's the middle element)
            # and no other element can appear more than once
            
            # Since the subsequence is of size 5, and mid is the middle element, it appears once
            # So, the other 4 elements must all be distinct and appear at most once
            # So, we need to choose 2 distinct elements from the less and 2 distinct elements from the greater
            # and ensure that no element appears more than once in the subsequence
            
            # The number of ways to choose 2 distinct elements from the less is C(less, 2)
            # Similarly for the greater
            ways_less = comb(less, 2)
            ways_greater = comb(greater, 2)
            
            # The total number of ways is ways_less * ways_greater * freq[mid]
            # But since mid is fixed as the middle element, and it appears once, we multiply by freq[mid]
            # because we have freq[mid] choices for the middle element
            total = (ways_less * ways_greater * freq[mid]) % MOD
            result = (result + total) % MOD
        
        return result