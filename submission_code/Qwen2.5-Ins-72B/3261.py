from typing import List

class Solution:
    def minOrAfterOperations(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_bit = 29  # Since 0 <= nums[i] < 2^30, the maximum bit position is 29
        result = 0
        
        for bit in range(max_bit, -1, -1):
            target = result | ((1 << bit) - 1)
            current = (1 << 30) - 1
            operations = 0
            
            for num in nums:
                current &= num
                if current | target == target:
                    current = (1 << 30) - 1
                else:
                    operations += 1
                    if operations > k:
                        break
            
            if operations <= k:
                result |= (1 << bit)
        
        return result