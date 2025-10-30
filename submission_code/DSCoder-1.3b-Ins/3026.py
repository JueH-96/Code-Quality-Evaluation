class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        # Create a list to store the numbers
        nums = [i for i in range(1, n+1)]
        
        # Initialize the result
        result = 0
        
        # Iterate over the list
        for i in range(n):
            # If the sum of the current number and the remaining numbers is less than or equal to target
            if nums[i] + sum(nums[i+1:]) <= target:
                # Add the sum to the result
                result += nums[i] + sum(nums[i+1:])
                # Subtract the sum from the target
                target -= nums[i] + sum(nums[i+1:])
            else:
                # If the sum of the current number and the remaining numbers is greater than target
                # Subtract the current number from the target
                target -= nums[i]
                # If the target is less than or equal to 0
                if target <= 0:
                    # Break the loop
                    break
        
        # Return the result
        return result