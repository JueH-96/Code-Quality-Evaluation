from typing import List

class Solution:
    def countPartitions(self, nums: List[int]) -> int:
        """
        Counts the number of partitions where the difference between the sum of the left
        and right subarrays is even.

        A partition at index i (0 <= i < n - 1) splits the array into nums[0..i] (left)
        and nums[i+1..n-1] (right).

        Let S_L be the sum of the left subarray and S_R be the sum of the right subarray.
        The difference S_L - S_R is even if and only if S_L and S_R have the same parity.

        Let S_Total be the sum of the entire array. S_Total = S_L + S_R.
        If S_L and S_R have the same parity (Both Even or Both Odd), S_Total is Even.
        If S_L and S_R have different parities (One Even, One Odd), S_Total is Odd.

        Therefore, S_L and S_R have the same parity if and only if S_Total is Even.

        Consequently, the difference S_L - S_R is even if and only if S_Total is Even.

        If the total sum of the array is even, then for *every* valid partition index i
        (from 0 to n-2), the sum of the left part and the sum of the right part will
        necessarily have the same parity, resulting in an even difference.
        The number of valid partition indices is n - 1.

        If the total sum of the array is odd, then for *every* valid partition index i,
        the sum of the left part and the sum of the right part will necessarily have
        different parities, resulting in an odd difference.
        The number of partitions with an even difference is 0.

        Args:
            nums: The input integer array.

        Returns:
            The number of partitions with an even sum difference.
        """
        # Get the length of the array.
        n = len(nums)
        
        # Calculate the total sum of all elements in the array.
        total_sum = sum(nums)

        # Check the parity of the total sum.
        if total_sum % 2 == 0:
            # If the total sum is even, any partition results in an even sum difference.
            # The number of possible partitions is n - 1 (for indices 0 to n-2).
            return n - 1
        else:
            # If the total sum is odd, no partition results in an even sum difference.
            return 0