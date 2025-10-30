from typing import List

class Solution:
    def countSetBits(self, n: int) -> int:
        count = 0
        while (n > 0):
            n &= (n-1)
            count += 1
        return count
    
    def canSortArray(self, nums: List[int]) -> bool:
        set_bit_counts = [self.countSetBits(num) for num in nums]
        groups = {}
        for i in range(len(nums)):
            count = set_bit_counts[i]
            if count not in groups:
                groups[count] = []
            groups[count].append(nums[i])
        
        sorted_groups = {}
        for count in groups:
            sorted_groups[count] = sorted(groups[count])
            
        constructed_nums = []
        group_indices = {}
        for count in groups:
            group_indices[count] = 0
            
        for count in set_bit_counts:
            num_to_append = sorted_groups[count][group_indices[count]]
            constructed_nums.append(num_to_append)
            group_indices[count] += 1
            
        return constructed_nums == sorted(nums)