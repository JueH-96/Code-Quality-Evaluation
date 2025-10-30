class Solution:
    def isArraySpecial(self, nums: List[int]) -> bool:
        n = len(nums)
        
        # Constraints: 1 <= nums.length <= 100.
        # If n = 1, there are no adjacent pairs. 
        # The condition "every pair of its adjacent elements contains two numbers with different parity"
        # is vacuously true. Example 1 (nums = [1]) confirms this.
        if n <= 1: # This handles the case n = 1.
            return True
        
        # Iterate through all adjacent pairs.
        # The loop runs from i = 0 to n-2.
        # This covers pairs (nums[0], nums[1]), ..., (nums[n-2], nums[n-1]).
        for i in range(n - 1):
            # Get the parity of the current element nums[i].
            # parity_current will be 0 if nums[i] is even, 1 if nums[i] is odd.
            parity_current = nums[i] % 2
            
            # Get the parity of the next element nums[i+1].
            # parity_next will be 0 if nums[i+1] is even, 1 if nums[i+1] is odd.
            parity_next = nums[i+1] % 2
            
            # If the parities are the same, the condition for a special array is violated.
            if parity_current == parity_next:
                return False
                
        # If the loop completes, it means all adjacent pairs have different parities.
        # Therefore, the array is special.
        return True