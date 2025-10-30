class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Sort the array to see what the target arrangement is
        sorted_nums = sorted(nums)
        
        # Compare the bit counts of nums and sorted_nums at each index
        for original_val, sorted_val in zip(nums, sorted_nums):
            if bin(original_val).count('1') != bin(sorted_val).count('1'):
                # If at any position the bit counts differ, 
                # we cannot make that change because elements 
                # with different bit counts cannot pass each other.
                return False
        return True