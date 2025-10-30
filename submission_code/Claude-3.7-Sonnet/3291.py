class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        # Helper function to count set bits in a number
        def count_set_bits(n):
            return bin(n).count('1')
        
        # Find the final state after sorting each segment
        result = []
        i = 0
        while i < len(nums):
            # Find all consecutive elements with the same number of set bits
            bits = count_set_bits(nums[i])
            segment = []
            while i < len(nums) and count_set_bits(nums[i]) == bits:
                segment.append(nums[i])
                i += 1
            
            # Sort the current segment and add to the result
            result.extend(sorted(segment))
        
        # Check if the resulting array is sorted
        return result == sorted(nums)