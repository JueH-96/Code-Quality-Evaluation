class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Helper function to count set bits
        def count_bits(n):
            return bin(n).count('1')
        
        # Create groups of consecutive elements with same bit count
        groups = []
        i = 0
        n = len(nums)
        
        while i < n:
            bit_count = count_bits(nums[i])
            group = []
            
            # Collect all consecutive elements with same bit count
            while i < n and count_bits(nums[i]) == bit_count:
                group.append(nums[i])
                i += 1
            
            groups.append(group)
        
        # For each group, check if it can be sorted correctly
        # The key insight: elements in a group can be rearranged among themselves,
        # but they cannot pass elements from other groups
        
        # Check if max of previous groups <= min of current group
        for i in range(1, len(groups)):
            max_prev = max(groups[i-1])
            min_curr = min(groups[i])
            
            if max_prev > min_curr:
                return False
        
        return True