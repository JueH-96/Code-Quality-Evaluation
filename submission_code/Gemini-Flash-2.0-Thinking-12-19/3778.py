from typing import List

class Solution:
    def transformArray(self, nums: List[int]) -> List[int]:
        # Step 1 & 2: Replace each even number with 0 and each odd number with 1.
        # Create a new list where each element is the transformation of the original element.
        # If x is even (x % 2 == 0), replace it with 0.
        # If x is odd, replace it with 1.
        transformed_nums = [0 if x % 2 == 0 else 1 for x in nums]

        # Step 3: Sort the modified array in non-decreasing order.
        # The sort() method sorts the list in-place.
        transformed_nums.sort()

        # Return the final sorted array.
        return transformed_nums