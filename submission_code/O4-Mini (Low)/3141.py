from typing import List

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total = sum(nums)
        # Number of full cycles we must take
        m = target // total
        rem = target - m * total
        
        # If there's no remainder, we just take m full cycles
        if rem == 0:
            # But if m == 0 and target == 0, smallest is 0 length (though target>=1 by constraints)
            return m * n
        
        # Now we need to find the shortest subarray in the infinite array
        # whose sum is rem. Since all nums[i] > 0, any minimal window
        # occurs within at most two consecutive copies of nums.
        arr = nums + nums
        best = float('inf')
        cur_sum = 0
        left = 0
        
        # Sliding window on arr, looking for sum == rem
        for right in range(len(arr)):
            cur_sum += arr[right]
            # Shrink from left while we exceed rem
            while left <= right and cur_sum > rem:
                cur_sum -= arr[left]
                left += 1
            if cur_sum == rem:
                length = right - left + 1
                # We only consider windows of length <= 2*n
                if length <= 2 * n:
                    best = min(best, length)
        
        if best == float('inf'):
            return -1
        
        return m * n + best