class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # The key observation is that the operation can be used to reduce the array
        # to a state where all elements are the same or zero.
        # If all elements are the same, the minimum length is 1.
        # If there are zeros, they cannot be reduced further, so the minimum length
        # is the number of zeros.

        # We will use a set to track unique non-zero elements.
        unique_non_zero = set()

        for num in nums:
            if num > 0:
                unique_non_zero.add(num)

        # If there are any non-zero elements, the minimum length is 1
        # because we can reduce all non-zero elements to the same value.
        # Otherwise, the minimum length is the number of zeros.
        return 1 if unique_non_zero else len(nums)