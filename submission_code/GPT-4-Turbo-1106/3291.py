class Solution:
    def canSortArray(self, nums: list[int]) -> bool:
        # Count the number of set bits for each number
        bit_counts = [bin(num).count('1') for num in nums]
        
        # Pair each number with its bit count
        paired = list(zip(bit_counts, nums))
        
        # Sort the paired list by bit count and then by the number itself
        sorted_paired = sorted(paired)
        
        # Check if the sorted paired list is the same as the original paired list
        return paired == sorted_paired