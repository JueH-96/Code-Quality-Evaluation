import math
from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        """
        Constructs a transformed array based on circular movements specified by elements in the input array.

        Args:
            nums: An integer array representing a circular array and movement steps.

        Returns:
            A new array where each element result[i] is the value from nums at the index
            reached by moving nums[i] steps (right if positive, left if negative) from index i
            in the circular array nums. If nums[i] is 0, result[i] is 0.
        """
        n = len(nums)
        if n == 0:
            return []  # Handle edge case of empty input

        result = [0] * n  # Initialize the result array with the same size

        for i in range(n):
            steps = nums[i]

            # The problem statement implies that if nums[i] == 0, result[i] should be 0.
            # However, let's check if the general movement logic also covers this.
            # If steps == 0, dest_index = (i + 0) % n = i % n = i.
            # Then result[i] would be nums[dest_index] = nums[i] = 0.
            # So, the general logic works for steps == 0 as well.

            # Calculate the destination index after moving 'steps' steps from index 'i'.
            # The modulo operator (%) handles the circular wrap-around correctly in Python,
            # even for negative steps (left movement).
            # For example:
            # If i=1, steps=-2, n=4: (1 + (-2)) % 4 = -1 % 4 = 3
            # If i=0, steps=-1, n=3: (0 + (-1)) % 3 = -1 % 3 = 2
            # If i=3, steps=3, n=4:  (3 + 3) % 4 = 6 % 4 = 2  <- Wait, example 1 says nums[0]=3 moves to nums[3], index 1? Let's re-read.
            
            # --- Re-reading Example 1 ---
            # Input: nums = [3,-2,1,1], n=4
            # nums[0] = 3: Start at index 0, move 3 steps right -> 1 -> 2 -> 3. Land at index 3. result[0] = nums[3] = 1.  (My initial calculation was wrong, (0+3)%4 = 3, which is correct index).
            # nums[1] = -2: Start at index 1, move 2 steps left -> 0 -> 3. Land at index 3. result[1] = nums[3] = 1. (My initial calculation (1-2)%4 = 3 is correct).
            # nums[2] = 1: Start at index 2, move 1 step right -> 3. Land at index 3. result[2] = nums[3] = 1. ((2+1)%4 = 3 is correct).
            # nums[3] = 1: Start at index 3, move 1 step right -> 0. Land at index 0. result[3] = nums[0] = 3. ((3+1)%4 = 0 is correct).
            # Output: [1, 1, 1, 3]. Okay, the formula (i + steps) % n seems correct. Let's recheck example 2.

            # --- Re-reading Example 2 ---
            # Input: nums = [-1,4,-1], n=3
            # nums[0] = -1: Start at index 0, move 1 left -> 2. Land at index 2. result[0] = nums[2] = -1. ((0-1)%3 = 2 is correct).
            # nums[1] = 4: Start at index 1, move 4 right -> 2 -> 0 -> 1 -> 2. Land at index 2. result[1] = nums[2] = -1. ((1+4)%3 = 5%3 = 2 is correct).
            # nums[2] = -1: Start at index 2, move 1 left -> 1. Land at index 1. result[2] = nums[1] = 4. ((2-1)%3 = 1 is correct).
            # Output: [-1,-1,4]. Okay, the formula (i + steps) % n seems robust.

            dest_index = (i + steps) % n
            
            # Set result[i] to the value at the destination index in the original nums array
            result[i] = nums[dest_index]

        return result