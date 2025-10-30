from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum operations to make all elements >= k
        by repeatedly removing the smallest element.

        Args:
            nums: A list of integers.
            k: The target integer value.

        Returns:
            The minimum number of operations.
        """
        # The problem requires removing elements less than k.
        # The operation is always removing the *smallest* element.
        # To make all elements >= k, we must remove any element that is currently < k.
        # Since we always remove the smallest, any element that is currently < k
        # will eventually either be removed because it is the smallest, or
        # other smaller elements will be removed until it becomes the smallest
        # among the remaining elements, at which point it will be removed
        # if it's still less than k.
        # Therefore, the total number of operations needed is simply the count
        # of elements in the initial array that are strictly less than k.

        operations_count = 0
        for num in nums:
            if num < k:
                operations_count += 1

        return operations_count