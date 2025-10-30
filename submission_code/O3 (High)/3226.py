from typing import List

class Solution:
    def numberGame(self, nums: List[int]) -> List[int]:
        # Sort the numbers so that every "minimum removal" is processed in order
        nums.sort()
        
        arr = []
        # Process the numbers in pairs: (smallest, second smallest)
        # Bob appends second-smallest first, then Alice appends the smallest
        for i in range(0, len(nums), 2):
            arr.append(nums[i + 1])  # Bob's element
            arr.append(nums[i])      # Alice's element
            
        return arr