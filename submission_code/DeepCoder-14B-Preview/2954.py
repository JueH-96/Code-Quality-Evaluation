from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        if k > n:
            return 0
        
        # Compute the prefix sum array
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + nums[i]
        
        max_sum = 0
        freq = defaultdict(int)
        unique = 0
        
        # Initialize the frequency dictionary for the first window
        for i in range(k):
            num = nums[i]
            if freq[num] == 0:
                unique += 1
            freq[num] += 1
        
        # Calculate the sum for the first window
        current_sum = prefix[k] - prefix[0]
        if unique >= m:
            max_sum = current_sum
        
        # Slide the window through the rest of the array
        for s in range(1, n - k + 1):
            # Remove the element leaving the window
            leaving = nums[s - 1]
            freq[leaving] -= 1
            if freq[leaving] == 0:
                unique -= 1
            
            # Add the new element entering the window
            entering = nums[s + k - 1]
            if freq[entering] == 0:
                unique += 1
            freq[entering] += 1
            
            # Calculate the current window's sum
            current_sum = prefix[s + k] - prefix[s]
            if unique >= m and current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum