class Solution:
    def findKOr(self, nums: List[int], k: int) -> int:
        if k == 1:
            # If k is 1, result is the bitwise OR of all elements
            result = 0
            for num in nums:
                result |= num
            return result
        
        if k == len(nums):
            # If k equals the length of nums, result is the bitwise AND of all elements
            result = nums[0]
            for num in nums[1:]:
                result &= num
            return result
        
        # General case: count how many times each bit position is set across all numbers
        bit_count = [0] * 32  # to count set bits up to 31st bit
        for num in nums:
            for i in range(32):
                if num & (1 << i):
                    bit_count[i] += 1
        
        # Calculate the K-or result
        result = 0
        for i in range(32):
            if bit_count[i] >= k:
                result |= (1 << i)
        
        return result