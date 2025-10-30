class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Function to count the number of set bits in a number
        def count_set_bits(n):
            return bin(n).count('1')
        
        # Create a list of tuples containing the number and its set bit count
        num_bits = [(num, count_set_bits(num)) for num in nums]
        
        # Group numbers by their set bit count
        from collections import defaultdict
        groups = defaultdict(list)
        for num, bits in num_bits:
            groups[bits].append(num)
        
        # Sort each group individually
        for key in groups:
            groups[key].sort()
        
        # Reconstruct the array by placing the sorted groups in their original order
        sorted_nums = []
        for num, bits in num_bits:
            sorted_nums.append(groups[bits].pop(0))
        
        # Check if the reconstructed array is sorted
        return sorted_nums == sorted(nums)