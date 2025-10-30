class Solution:
    def canSortArray(self, nums: List[int]) -> bool:
        def count_set_bits(n):
            return bin(n).count('1')
        
        prev_max = 0
        i = 0
        
        while i < len(nums):
            # Find segment of consecutive elements with same set bit count
            j = i
            set_bits = count_set_bits(nums[i])
            while j < len(nums) and count_set_bits(nums[j]) == set_bits:
                j += 1
            
            # Sort the segment
            segment = sorted(nums[i:j])
            
            # Check if segment can be placed after previous segments
            if segment[0] < prev_max:
                return False
            
            prev_max = segment[-1]
            i = j
        
        return True