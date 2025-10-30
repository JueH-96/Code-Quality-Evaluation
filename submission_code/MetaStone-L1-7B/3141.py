import bisect

class Solution:
    def minSizeSubarray(self, nums: List[int], target: int) -> int:
        # Compute the prefix sum array.
        prefix = [0]
        for num in nums:
            prefix.append(prefix[-1] + num)
        S = prefix[-1]
        n = len(nums)
        min_len = float('inf')
        
        # Case 1: subarray within the first cycle.
        left = 0
        current_sum = 0
        for right in range(n):
            current_sum += nums[right]
            while current_sum > target and left <= right:
                current_sum -= nums[left]
                left += 1
            if current_sum == target:
                min_len = min(min_len, right - left + 1)
        
        # Case 2: subarray that wraps around.
        min_case2 = float('inf')
        for k in range(n):
            if k == 0:
                s = 0
            else:
                s = prefix[-1] - prefix[-k-1]
            if s > target:
                continue
            needed = target - s
            if needed < 0:
                continue
            if needed > S:
                continue
            # Binary search for needed in prefix.
            idx = bisect.bisect_left(prefix, needed)
            if idx < len(prefix) and prefix[idx] == needed:
                m = idx
                total_length = k + m
                if total_length < min_case2:
                    min_case2 = total_length
        
        # Determine the minimal length.
        if min_len == float('inf') and min_case2 == float('inf'):
            return -1
        else:
            result = min(min_len, min_case2)
            return result if result != float('inf') else -1