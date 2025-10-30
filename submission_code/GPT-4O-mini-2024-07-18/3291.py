class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        from collections import defaultdict
        
        # Function to count the number of set bits in a number
        def count_set_bits(x):
            return bin(x).count('1')
        
        # Create a mapping of set bits to the corresponding numbers
        set_bits_map = defaultdict(list)
        for num in nums:
            set_bits_map[count_set_bits(num)].append(num)
        
        # Check if we can sort each group of numbers with the same set bits
        for group in set_bits_map.values():
            if sorted(group) != group:
                return False
        
        return True