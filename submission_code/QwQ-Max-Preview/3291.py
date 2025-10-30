class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        if not nums:
            return True  # As per constraints, nums is non-empty
        
        # Split into groups of consecutive elements with the same number of set bits
        groups = []
        current_bit = None
        current_group = []
        for num in nums:
            bits = bin(num).count('1')
            if bits != current_bit:
                if current_group:
                    groups.append(current_group)
                current_group = [num]
                current_bit = bits
            else:
                current_group.append(num)
        if current_group:
            groups.append(current_group)
        
        # Sort each group and check the max of each group with the min of the next
        sorted_groups = [sorted(g) for g in groups]
        for i in range(len(sorted_groups) - 1):
            if sorted_groups[i][-1] > sorted_groups[i+1][0]:
                return False
        return True