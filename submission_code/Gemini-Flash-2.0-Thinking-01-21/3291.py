from typing import List

class Solution:
    def count_set_bits(self, n: int) -> int:
        count = 0
        while n > 0:
            n &= (n - 1)
            count += 1
        return count

    def canSortArray(self, nums: List[int]) -> bool:
        set_bit_counts = [self.count_set_bits(num) for num in nums]
        groups = {}
        for i in range(len(nums)):
            bits = set_bit_counts[i]
            if bits not in groups:
                groups[bits] = []
            groups[bits].append(nums[i])
        for bits in groups:
            groups[bits].sort()
        
        constructed_nums = []
        group_indices = {}
        for bits in groups:
            group_indices[bits] = 0
            
        for i in range(len(nums)):
            bits = set_bit_counts[i]
            constructed_nums.append(groups[bits][group_indices[bits]])
            group_indices[bits] += 1
            
        return constructed_nums == sorted(nums)