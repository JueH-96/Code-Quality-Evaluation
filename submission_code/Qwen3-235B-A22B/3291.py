class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_bits(n):
            return bin(n).count('1')
        
        # Split into contiguous groups with same bit count
        groups = []
        if not nums:
            return True
        current_bit_count = count_bits(nums[0])
        current_group = [nums[0]]
        
        for i in range(1, len(nums)):
            bit_count = count_bits(nums[i])
            if bit_count == current_bit_count:
                current_group.append(nums[i])
            else:
                groups.append(current_group)
                current_group = [nums[i]]
                current_bit_count = bit_count
        groups.append(current_group)  # add the last group
        
        # Sort each group
        for group in groups:
            group.sort()
        
        # Concatenate the groups and check if the result is sorted
        sorted_array = []
        for group in groups:
            sorted_array.extend(group)
        
        for i in range(len(sorted_array) - 1):
            if sorted_array[i] > sorted_array[i+1]:
                return False
        return True