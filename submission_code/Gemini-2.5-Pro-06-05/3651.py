from typing import List

class Solution:
    def constructTransformedArray(self, nums: List[int]) -> List[int]:
        """
        Constructs a new array based on circular movements determined by the values in the input array.

        Args:
            nums: A list of integers representing a circular array.

        Returns:
            A new list where each element is determined by the rules of circular movement.
        """
        n = len(nums)

        # A list comprehension provides a concise and efficient way to build the result array.
        # We iterate through `nums` with `enumerate` to get both the index `i` and value `val`.
        #
        # For each element:
        # - A conditional expression checks if `val` is zero.
        # - If `val` is 0, the result for this position is 0.
        # - If `val` is not 0, we calculate the destination index using the modulo operator.
        #   The expression `(i + val) % n` handles circular wrapping for both positive (right)
        #   and negative (left) movements correctly in Python. The result for this position
        #   is the value from the original `nums` array at this destination index.
        return [nums[(i + val) % n] if val != 0 else 0 for i, val in enumerate(nums)]