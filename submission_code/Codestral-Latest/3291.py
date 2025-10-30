class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(n):
            return bin(n).count('1')

        # Create a list of tuples where each tuple is (number, count of set bits)
        nums_with_bits = [(num, count_set_bits(num)) for num in nums]

        # Sort the list by the count of set bits
        nums_with_bits.sort(key=lambda x: x[1])

        # Extract the sorted numbers
        sorted_nums = [num for num, bits in nums_with_bits]

        # Check if the sorted numbers are the same as the original sorted nums
        return sorted_nums == sorted(nums)