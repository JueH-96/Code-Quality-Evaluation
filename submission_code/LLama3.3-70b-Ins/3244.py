from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        """
        This function calculates the minimum length of the array after performing the given operations.
        
        The idea is to keep removing the largest and the second largest numbers from the array until we are left with one or two numbers.
        
        :param nums: A list of positive integers.
        :return: The minimum length of the array after performing the operations.
        """
        
        # Sort the array in descending order
        nums.sort(reverse=True)
        
        # Initialize the result (minimum length) as the length of the array
        result = len(nums)
        
        # Iterate over the array
        while len(nums) > 1:
            # If the largest number is greater than the second largest number, 
            # we can remove both of them and add their remainder to the array
            if nums[0] > nums[1]:
                # Calculate the remainder
                remainder = nums[0] % nums[1]
                # Remove the largest and the second largest numbers from the array
                nums = nums[2:]
                # If the remainder is not zero, add it to the array
                if remainder != 0:
                    nums.append(remainder)
                # Sort the array in descending order
                nums.sort(reverse=True)
            # If the largest number is not greater than the second largest number, 
            # we can remove the largest number from the array
            else:
                # Remove the largest number from the array
                nums = nums[1:]
        
        # Update the result (minimum length)
        result = len(nums)
        
        return result