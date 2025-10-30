class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        from collections import defaultdict
        
        def count_set_bits(x):
            return bin(x).count('1')
        
        # Group numbers by their number of set bits
        bit_count_groups = defaultdict(list)
        for num in nums:
            bit_count = count_set_bits(num)
            bit_count_groups[bit_count].append(num)
        
        # Check if each group can be sorted individually
        for key in bit_count_groups:
            bit_count_groups[key].sort()
        
        # Reconstruct the array from the sorted groups
        sorted_nums = []
        for num in nums:
            bit_count = count_set_bits(num)
            sorted_nums.append(bit_count_groups[bit_count].pop(0))
        
        # Check if the reconstructed array is sorted
        return sorted_nums == sorted(nums)