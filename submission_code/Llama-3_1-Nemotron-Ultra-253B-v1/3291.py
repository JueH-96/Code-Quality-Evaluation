from typing import List

class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_bits(n: int) -> int:
            return bin(n).count('1')
        
        def get_groups(arr: List[int]) -> List[List[int]]:
            groups = []
            if not arr:
                return groups
            current_bit = count_bits(arr[0])
            current_group = [arr[0]]
            for num in arr[1:]:
                bit = count_bits(num)
                if bit == current_bit:
                    current_group.append(num)
                else:
                    groups.append(current_group)
                    current_bit = bit
                    current_group = [num]
            groups.append(current_group)
            return groups
        
        sorted_nums = sorted(nums)
        groups_original = get_groups(nums)
        groups_sorted = get_groups(sorted_nums)
        
        # Check if group sequences match
        group_seq_original = [count_bits(group[0]) for group in groups_original]
        group_seq_sorted = [count_bits(group[0]) for group in groups_sorted]
        
        if group_seq_original != group_seq_sorted:
            return False
        
        # Check each corresponding group's elements
        for og, sg in zip(groups_original, groups_sorted):
            if sorted(og) != sg:
                return False
        
        return True