class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        from collections import defaultdict
        
        # Function to count the number of set bits in an integer
        def count_set_bits(x):
            return bin(x).count('1')
        
        # Dictionary to store lists of numbers grouped by their set bit counts
        bit_groups = defaultdict(list)
        
        # Populate the dictionary with numbers grouped by their set bit counts
        for num in nums:
            bit_groups[count_set_bits(num)].append(num)
        
        # For each group, sort the numbers
        for key in bit_groups:
            bit_groups[key].sort()
        
        # Reconstruct the array from the sorted groups
        sorted_nums = []
        for key in sorted(bit_groups.keys()):
            sorted_nums.extend(bit_groups[key])
        
        # Check if the reconstructed sorted array matches the globally sorted array
        return sorted_nums == sorted(nums)