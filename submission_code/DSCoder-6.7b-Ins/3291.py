class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_bits(n):
            return bin(n).count('1')

        sorted_nums = sorted(nums, key=count_bits)
        return nums == sorted_nums