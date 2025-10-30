class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        # Create an array to keep track of the total decrements needed
        total_decrements = [0] * (len(nums) + 1)
        
        # Process each query to mark the increments and decrements
        for l, r in queries:
            total_decrements[l] += 1
            if r + 1 < len(total_decrements):
                total_decrements[r + 1] -= 1
        
        # Calculate the actual decrements for each index
        current_decrement = 0
        for i in range(len(nums)):
            current_decrement += total_decrements[i]
            # Check if the current number can be reduced to zero
            if nums[i] < current_decrement:
                return False
        
        return True