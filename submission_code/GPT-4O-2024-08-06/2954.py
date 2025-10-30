class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        from collections import defaultdict
        
        n = len(nums)
        max_sum = 0
        current_sum = 0
        distinct_count = 0
        freq = defaultdict(int)
        
        # Initialize the first window
        for i in range(k):
            current_sum += nums[i]
            if freq[nums[i]] == 0:
                distinct_count += 1
            freq[nums[i]] += 1
        
        # Check if the first window is almost unique
        if distinct_count >= m:
            max_sum = current_sum
        
        # Slide the window
        for i in range(k, n):
            # Remove the element going out of the window
            outgoing = nums[i - k]
            freq[outgoing] -= 1
            if freq[outgoing] == 0:
                distinct_count -= 1
            current_sum -= outgoing
            
            # Add the new element coming into the window
            incoming = nums[i]
            if freq[incoming] == 0:
                distinct_count += 1
            freq[incoming] += 1
            current_sum += incoming
            
            # Check if the current window is almost unique
            if distinct_count >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum