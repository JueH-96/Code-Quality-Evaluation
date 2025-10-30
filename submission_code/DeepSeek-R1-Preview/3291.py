class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        if not nums:
            return True
        
        # Split into groups of consecutive elements with the same number of set bits
        groups = []
        current_group = []
        current_bits = None
        for num in nums:
            bits = bin(num).count('1')
            if bits != current_bits:
                if current_group:
                    groups.append(current_group)
                current_group = [num]
                current_bits = bits
            else:
                current_group.append(num)
        if current_group:
            groups.append(current_group)
        
        # Sort each group
        sorted_groups = [sorted(group) for group in groups]
        
        # Concatenate the sorted groups
        sorted_nums = []
        for sg in sorted_groups:
            sorted_nums.extend(sg)
        
        # Check if the concatenated array is sorted
        for i in range(1, len(sorted_nums)):
            if sorted_nums[i] < sorted_nums[i-1]:
                return False
        return True