from typing import List

class Solution:
    def countPairs(self, nums: List[int]) -> int:
        
        def can_swap_to_equal(num1: int, num2: int) -> bool:
            """
            Checks if swapping exactly two digits in num1 can result in num2.
            Args:
                num1: The first integer.
                num2: The second integer.
            Returns:
                True if a single swap in num1 results in num2, False otherwise.
            """
            s1 = str(num1)
            n = len(s1)
            
            # If num1 has only one digit, cannot perform a swap of two distinct digits
            if n < 2:
                return False

            # Iterate through all distinct pairs of indices (i, j) in the string
            for i in range(n):
                for j in range(i + 1, n):
                    # Create the new string by swapping digits at i and j
                    s1_list = list(s1)
                    s1_list[i], s1_list[j] = s1_list[j], s1_list[i]
                    s1_swapped = "".join(s1_list)
                    
                    # Convert the swapped string back to integer and compare with num2
                    # int() handles leading zeros correctly (e.g., int("03") is 3)
                    if int(s1_swapped) == num2:
                        return True
                        
            # No single swap resulted in num2
            return False

        def are_almost_equal(x: int, y: int) -> bool:
            """
            Checks if two integers x and y are almost equal according to the problem definition.
            Args:
                x: The first integer.
                y: The second integer.
            Returns:
                True if x and y are almost equal, False otherwise.
            """
            # Case 1: The numbers are already equal (0 swaps needed)
            if x == y:
                return True
            
            # Case 2: Swapping digits in x can make it equal to y (1 swap needed in x)
            if can_swap_to_equal(x, y):
                return True
                
            # Case 3: Swapping digits in y can make it equal to x (1 swap needed in y)
            # Note: checking can_swap_to_equal(x, y) and can_swap_to_equal(y, x) covers
            # the condition "perform the following operation at most once: Choose either x or y".
            # If they are equal, 0 swaps. If can_swap_to_equal(x,y) is true, 1 swap on x.
            # If can_swap_to_equal(y,x) is true, 1 swap on y.
            if can_swap_to_equal(y, x):
                return True
                
            # If none of the above conditions are met, they are not almost equal
            return False

        count = 0
        n = len(nums)
        
        # Iterate through all unique pairs of indices (i, j) such that i < j
        for i in range(n):
            for j in range(i + 1, n):
                # Check if the elements at these indices are almost equal
                if are_almost_equal(nums[i], nums[j]):
                    count += 1

        # Return the total count of almost equal pairs
        return count