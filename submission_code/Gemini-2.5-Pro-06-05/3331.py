from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum number of operations to make all elements in nums >= k.

        The operation is to remove the smallest element. This means we will always
        remove elements that are smaller than k until no such elements are left.
        Therefore, the number of operations is simply the count of elements in the
        initial array that are less than k.

        Args:
            nums: A list of integers.
            k: The target minimum value.

        Returns:
            The minimum number of operations.
        """
        
        operations_needed = 0
        for num in nums:
            if num < k:
                operations_needed += 1
        
        return operations_needed