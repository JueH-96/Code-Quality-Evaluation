from typing import List

class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        # An array with 0 or 1 element is always considered special
        # because there are no adjacent pairs to violate the condition.
        if len(nums) <= 1:
            return True
        
        # Iterate through the array up to the second-to-last element
        # to check all adjacent pairs (nums[i], nums[i+1]).
        for i in range(len(nums) - 1):
            num1 = nums[i]
            num2 = nums[i+1]
            
            # Get the parity of each number.
            # 0 indicates even, 1 indicates odd.
            parity1 = num1 % 2
            parity2 = num2 % 2
            
            # If the parities are the same, the array is not special.
            # We can immediately return False.
            if parity1 == parity2:
                return False
                
        # If the loop completes, it means all adjacent pairs had different parities.
        # Therefore, the array is special.
        return True