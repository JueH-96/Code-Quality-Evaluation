from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        """
        Calculates the minimum number of operations to make all elements in nums equal to k.
        """

        # The operation can only decrease values. If any number is already less than k,
        # it's impossible to make it equal to k.
        if any(num < k for num in nums):
            return -1

        # The minimum number of operations is the count of unique values that are
        # strictly greater than k. Each such unique value "level" must be reduced
        # in a separate operation. The optimal strategy is to reduce the current
        # largest value to the next largest value in the array, repeating until all
        # values are less than or equal to k.
        
        # We can find this count by creating a set of all numbers in `nums` that
        # are greater than k and then taking its size.
        return len(set(num for num in nums if num > k))