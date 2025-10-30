from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        def is_almost_equal(num1: int, num2: int) -> bool:
            """Check if two numbers are almost equal."""
            # Convert numbers to strings for easier manipulation
            str_num1, str_num2 = str(num1), str(num2)
            
            # If the numbers have different lengths, they can't be almost equal
            if len(str_num1) != len(str_num2):
                return False
            
            # Initialize a counter for the number of different digits
            diff_count = 0
            
            # Initialize a list to store the indices of different digits
            diff_indices = []
            
            # Iterate over the digits of the numbers
            for i in range(len(str_num1)):
                if str_num1[i] != str_num2[i]:
                    diff_count += 1
                    diff_indices.append(i)
                    
                    # If there are more than 2 different digits, the numbers can't be almost equal
                    if diff_count > 2:
                        return False
            
            # If there are 0 or 1 different digits, the numbers are already equal or can't be made equal
            if diff_count < 2:
                return diff_count == 0
            
            # If there are 2 different digits, check if swapping them makes the numbers equal
            return str_num1[diff_indices[0]] == str_num2[diff_indices[1]] and str_num1[diff_indices[1]] == str_num2[diff_indices[0]]
        
        # Initialize a counter for the number of almost equal pairs
        count = 0
        
        # Iterate over the list of numbers
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                # Check if the current pair of numbers is almost equal
                if is_almost_equal(nums[i], nums[j]):
                    count += 1
        
        return count