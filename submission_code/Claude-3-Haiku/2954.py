from collections import defaultdict

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        max_sum = 0
        
        # Create a dictionary to store the frequency of each element in the current window
        freq = defaultdict(int)
        
        # Initialize the current window
        for i in range(k):
            freq[nums[i]] += 1
        
        # Check if the current window is almost unique
        if len(freq) >= m:
            max_sum = sum(nums[:k])
        
        # Slide the window and update the maximum sum
        for i in range(k, n):
            # Remove the leftmost element from the window
            freq[nums[i - k]] -= 1
            if freq[nums[i - k]] == 0:
                del freq[nums[i - k]]
            
            # Add the rightmost element to the window
            freq[nums[i]] += 1
            
            # Check if the current window is almost unique
            if len(freq) >= m:
                max_sum = max(max_sum, sum(nums[i - k + 1:i + 1]))
        
        return max_sum