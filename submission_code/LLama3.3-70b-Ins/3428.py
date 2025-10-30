from typing import List

class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        # This function is not needed for the problem, we can solve it directly in the duplicateNumbersXOR function.
        pass

    def duplicateNumbersXOR(self, nums: List[int]) -> int:
        """
        This function calculates the bitwise XOR of all numbers that appear twice in the given list.
        
        Args:
        nums (List[int]): A list of integers where each number appears either once or twice.
        
        Returns:
        int: The bitwise XOR of all numbers that appear twice in the list, or 0 if no number appears twice.
        """
        
        # Initialize a dictionary to store the count of each number
        count_dict = {}
        
        # Initialize the result variable to 0
        result = 0
        
        # Iterate over each number in the list
        for num in nums:
            # If the number is already in the dictionary, it means it appears twice
            if num in count_dict:
                # XOR the result with the number
                result ^= num
            else:
                # If the number is not in the dictionary, add it
                count_dict[num] = 1
        
        # Return the result
        return result