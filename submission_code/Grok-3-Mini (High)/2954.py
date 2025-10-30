class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        # Initialize frequency map, distinct count, and current sum for the first window
        freq = {}
        distinct = 0
        current_sum = 0
        for i in range(k):
            current_sum += nums[i]
            if nums[i] not in freq:
                freq[nums[i]] = 0
            freq[nums[i]] += 1
            if freq[nums[i]] == 1:  # First occurrence
                distinct += 1
        
        # Initialize max_sum
        max_sum_val = 0
        if distinct >= m:
            max_sum_val = current_sum
        
        # Slide the window
        left = 0
        for _ in range(n - k):  # Number of slides
            # Remove the element sliding out
            remove_val = nums[left]
            freq[remove_val] -= 1
            if freq[remove_val] == 0:
                distinct -= 1
            current_sum -= remove_val
            
            # Add the new element sliding in
            add_val = nums[left + k]
            if add_val not in freq:
                freq[add_val] = 0
            freq[add_val] += 1
            if freq[add_val] == 1:  # Newly added distinct element
                distinct += 1
            current_sum += add_val
            
            # Move the left pointer
            left += 1
            
            # Check the new window
            if distinct >= m:
                max_sum_val = max(max_sum_val, current_sum)
        
        return max_sum_val