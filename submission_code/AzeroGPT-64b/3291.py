from collections import defaultdict

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(num):
            return bin(num).count('1')

        # Count set bits for each number and group them by the number of set bits
        bit_groups = defaultdict(list)
        for num in nums:
            bit_groups[count_set_bits(num)].append(num)

        # Sort each group
        for group in bit_groups.values():
            group.sort()

        # Concatenate sorted groups and check if it's the sorted nums
        sorted_nums = sum(list(bit_groups.values()), start=[])
        return sorted_nums == sorted(nums)