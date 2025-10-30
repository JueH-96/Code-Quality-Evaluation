class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Calculate the number of set bits for each element in the original array
        original_bit_count = [bin(x).count('1') for x in nums]

        # Sort the array
        sorted_nums = sorted(nums)
        # Calculate the number of set bits for each element in the sorted array
        sorted_bit_count = [bin(x).count('1') for x in sorted_nums]

        # We can only swap adjacent elements if they have the same number of set bits,
        # so the pattern of set-bits across positions must remain the same.
        # Therefore, we check if the bit-count pattern in the original array matches
        # the bit-count pattern in the sorted array.
        return original_bit_count == sorted_bit_count