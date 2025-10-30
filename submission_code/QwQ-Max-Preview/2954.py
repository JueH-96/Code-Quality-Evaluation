from typing import List

class Solution:
    def maxSum(self, nums: List[int], m: int, k: int) -> int:
        n = len(nums)
        max_sum = 0
        current_sum = 0
        freq = {}
        
        # Initialize the first window
        for i in range(k):
            num = nums[i]
            current_sum += num
            if num in freq:
                freq[num] += 1
            else:
                freq[num] = 1
        
        if len(freq) >= m:
            max_sum = current_sum
        
        # Slide the window through the rest of the array
        for i in range(k, n):
            left_num = nums[i - k]
            # Remove the leftmost element from the frequency dict
            freq[left_num] -= 1
            if freq[left_num] == 0:
                del freq[left_num]
            
            right_num = nums[i]
            # Add the new right element to the frequency dict
            if right_num in freq:
                freq[right_num] += 1
            else:
                freq[right_num] = 1
            
            # Update the current sum
            current_sum = current_sum - left_num + right_num
            
            # Check if the current window is valid and update max_sum
            if len(freq) >= m and current_sum > max_sum:
                max_sum = current_sum
        
        return max_sum