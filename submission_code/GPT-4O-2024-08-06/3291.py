class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        from collections import defaultdict
        
        # Helper function to count the number of set bits in a number
        def count_set_bits(x):
            return bin(x).count('1')
        
        # Create a dictionary to group numbers by their set bit count
        bit_count_groups = defaultdict(list)
        
        # Populate the dictionary
        for num in nums:
            bit_count = count_set_bits(num)
            bit_count_groups[bit_count].append(num)
        
        # For each group, check if the numbers can be sorted
        for group in bit_count_groups.values():
            if group != sorted(group):
                return False
        
        return True