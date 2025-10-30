class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        if not nums:
            return True
        
        # Calculate the number of set bits for each number
        set_bits = [bin(num).count('1') for num in nums]
        
        # Group numbers into contiguous segments with the same set bits
        groups = []
        current_group = [nums[0]]
        current_bit = set_bits[0]
        
        for i in range(1, len(nums)):
            if set_bits[i] == current_bit:
                current_group.append(nums[i])
            else:
                groups.append(current_group)
                current_group = [nums[i]]
                current_bit = set_bits[i]
        groups.append(current_group)  # Add the last group
        
        # Sort each group and concatenate
        sorted_groups = [sorted(group) for group in groups]
        concatenated = []
        for group in sorted_groups:
            concatenated.extend(group)
        
        # Check if the concatenated result is the sorted array
        return concatenated == sorted(nums)