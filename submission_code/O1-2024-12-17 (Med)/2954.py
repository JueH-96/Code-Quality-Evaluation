class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        from collections import defaultdict
        
        freq = defaultdict(int)    # frequency map for elements in the current window
        distinct_count = 0         # how many distinct elements are in the current window
        window_sum = 0             # sum of the current window
        max_sum = 0                # track maximum sum of valid windows
        
        left = 0
        for right in range(len(nums)):
            # Add the new element on the right
            freq[nums[right]] += 1
            if freq[nums[right]] == 1:
                distinct_count += 1
            window_sum += nums[right]
            
            # If window size exceeds k, shrink from the left
            if right - left + 1 > k:
                freq[nums[left]] -= 1
                if freq[nums[left]] == 0:
                    distinct_count -= 1
                window_sum -= nums[left]
                left += 1
            
            # Check if the current window is exactly size k
            if right - left + 1 == k:
                # If we have at least m distinct elements, update max_sum
                if distinct_count >= m:
                    max_sum = max(max_sum, window_sum)
        
        return max_sum