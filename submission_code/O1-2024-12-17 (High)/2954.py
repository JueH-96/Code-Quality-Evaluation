class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        from collections import defaultdict
        
        freq = defaultdict(int)
        window_sum = 0
        distinct_count = 0
        ans = 0
        
        # Initialize the first window of size k
        for i in range(k):
            if freq[nums[i]] == 0:
                distinct_count += 1
            freq[nums[i]] += 1
            window_sum += nums[i]
        
        # Check if the first window is almost unique
        if distinct_count >= m:
            ans = max(ans, window_sum)
        
        # Slide the window from index k onwards
        for i in range(k, len(nums)):
            left_val = nums[i - k]
            freq[left_val] -= 1
            if freq[left_val] == 0:
                distinct_count -= 1
            window_sum -= left_val
            
            if freq[nums[i]] == 0:
                distinct_count += 1
            freq[nums[i]] += 1
            window_sum += nums[i]
            
            # Update the answer if this window is almost unique
            if distinct_count >= m:
                ans = max(ans, window_sum)
        
        return ans