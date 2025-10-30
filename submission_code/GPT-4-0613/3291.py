class Solution:
    def canSortArray(self, nums):
        bit_counts = [bin(n).count('1') for n in nums]
        return bit_counts == sorted(bit_counts)