from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        if not nums or numOperations == 0:
            return 0
        
        nums.sort()
        n = len(nums)
        max_freq = 1
        left = 0
        total = 0
        
        for right in range(n):
            target = nums[right]
            total += target
            
            # Calculate the number of operations needed to make all elements from left to right equal to target
            # The number of operations is (right - left + 1) * target - (sum of nums[left..right])
            # Since we can perform at most numOperations operations, we need to ensure that the required operations <= numOperations * k
            # However, since each operation can add any value in [-k, k], the total possible change per operation is 2k (from -k to +k)
            # But to maximize frequency, we should try to make as many elements as possible equal to a target.
            # So, we need to find the largest window where the sum of differences can be covered by the allowed operations.
            
            # The required operations to make all elements in the window equal to target is:
            # (right - left + 1) * target - (sum of nums[left..right])
            # Since each operation can change an element by at most k (in absolute value), the total possible change is numOperations * k
            # So, we need to find the largest window where (right - left + 1) * target - sum <= numOperations * k
            
            while (right - left + 1) * target - total > numOperations * k:
                total -= nums[left]
                left += 1
            
            max_freq = max(max_freq, right - left + 1)
        
        return max_freq