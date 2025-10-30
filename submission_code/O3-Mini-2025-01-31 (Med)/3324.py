from collections import Counter
from typing import List

class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        # Count the frequency of each number in the array.
        freq = Counter(nums)
        
        # For each number, if it appears more than twice, then one group will have to get a duplicate.
        # Hence, it's impossible to split the array into two parts that both only contain distinct elements.
        for count in freq.values():
            if count > 2:
                return False
        
        # If no number appears more than twice, then the forced assignments for numbers that appear twice
        # (one in each group) and the free assignment of numbers that appear once will always allow a valid split.
        return True

# Sample usage:
# sol = Solution()
# print(sol.isPossibleToSplit([1,1,2,2,3,4]))  # Expected output: True
# print(sol.isPossibleToSplit([1,1,1,1]))        # Expected output: False