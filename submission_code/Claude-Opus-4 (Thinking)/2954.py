class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        if len(nums) < k:
            return 0
        
        from collections import defaultdict
        count = defaultdict(int)
        current_sum = 0
        distinct_count = 0
        
        # Build the first window
        for i in range(k):
            if count[nums[i]] == 0:
                distinct_count += 1
            count[nums[i]] += 1
            current_sum += nums[i]
        
        # Check if the first window is almost unique
        max_sum = 0
        if distinct_count >= m:
            max_sum = current_sum
        
        # Slide the window
        for i in range(k, len(nums)):
            # Add the new element
            if count[nums[i]] == 0:
                distinct_count += 1
            count[nums[i]] += 1
            current_sum += nums[i]
            
            # Remove the leftmost element
            left_elem = nums[i - k]
            count[left_elem] -= 1
            if count[left_elem] == 0:
                distinct_count -= 1
            current_sum -= left_elem
            
            # Check if current window is almost unique
            if distinct_count >= m:
                max_sum = max(max_sum, current_sum)
        
        return max_sum