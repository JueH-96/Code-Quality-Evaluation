class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        n = len(nums)
        total_sum = sum(nums)
        
        # Calculate how many complete repetitions of nums we need
        quotient, remainder = divmod(target, total_sum)
        
        # If remainder is 0, we just need complete repetitions
        if remainder == 0:
            return quotient * n
        
        # Double the array to handle wrap-around cases
        nums2 = nums * 2
        
        # Find shortest subarray with sum equal to remainder or target
        min_length_remainder = float('inf')
        min_length_target = float('inf')
        
        # Sliding window for remainder
        window_sum = 0
        left = 0
        
        for right in range(len(nums2)):
            window_sum += nums2[right]
            
            while window_sum > remainder and left <= right:
                window_sum -= nums2[left]
                left += 1
            
            if window_sum == remainder:
                min_length_remainder = min(min_length_remainder, right - left + 1)
        
        # Sliding window for target (in case target < total_sum)
        window_sum = 0
        left = 0
        
        for right in range(len(nums2)):
            window_sum += nums2[right]
            
            while window_sum > target and left <= right:
                window_sum -= nums2[left]
                left += 1
            
            if window_sum == target:
                min_length_target = min(min_length_target, right - left + 1)
        
        # Calculate final result
        if min_length_remainder != float('inf'):
            result1 = quotient * n + min_length_remainder
        else:
            result1 = float('inf')
        
        result2 = min_length_target
        
        # Choose the minimum length
        result = min(result1, result2)
        
        return result if result != float('inf') else -1