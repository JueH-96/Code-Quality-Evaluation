class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(n):
            return bin(n).count('1')
        
        # Create groups of consecutive elements with same set bit count
        groups = []
        current_group = [nums[0]]
        current_bits = count_set_bits(nums[0])
        
        for i in range(1, len(nums)):
            bits = count_set_bits(nums[i])
            if bits == current_bits:
                current_group.append(nums[i])
            else:
                groups.append(current_group)
                current_group = [nums[i]]
                current_bits = bits
        
        groups.append(current_group)
        
        # Sort each group individually
        result = []
        for group in groups:
            result.extend(sorted(group))
        
        # Check if result matches the fully sorted array
        return result == sorted(nums)