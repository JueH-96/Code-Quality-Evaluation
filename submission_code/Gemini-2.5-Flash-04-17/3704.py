from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        """
        Counts the number of partitions where the difference between the sum
        of the left and right subarrays is even.
        A partition is defined by an index i where 0 <= i < n - 1, splitting
        the array into nums[0...i] and nums[i+1...n-1].
        """
        n = len(nums)
        
        # Calculate the total sum of the array.
        total_sum = sum(nums)
        
        # Let S_L be the sum of the left subarray (indices 0 to i)
        # Let S_R be the sum of the right subarray (indices i+1 to n-1)
        # The condition is that S_L - S_R is even.
        # We know that S_L + S_R = total_sum.
        # Substituting S_R = total_sum - S_L, the difference becomes:
        # S_L - (total_sum - S_L) = 2 * S_L - total_sum.
        # For this difference (2 * S_L - total_sum) to be even:
        # Since 2 * S_L is always even, the parity of (2 * S_L - total_sum)
        # is determined by the parity of -total_sum, which is the same as
        # the parity of total_sum.
        # Therefore, the difference S_L - S_R is even if and only if
        # total_sum is even.
        
        # If the total sum of the array is odd, then for any partition i,
        # the difference S_L - S_R will be odd.
        # In this case, there are no partitions that satisfy the condition.
        if total_sum % 2 != 0:
            return 0
        
        # If the total sum of the array is even, then for any partition i,
        # the difference S_L - S_R will be even.
        # We need to count the number of possible partitions.
        # A partition index i must satisfy 0 <= i < n - 1.
        # The possible integer values for i are 0, 1, ..., n-2.
        # For each such i, the left subarray nums[0...i] and the right subarray
        # nums[i+1...n-1] are guaranteed to be non-empty because n >= 2.
        # The number of possible partition indices i is (n-2) - 0 + 1 = n - 1.
        # Since the total sum is even, all these n-1 partitions satisfy the condition.
        return n - 1