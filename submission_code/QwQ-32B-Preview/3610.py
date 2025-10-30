from typing import List
from collections import Counter

class Solution:
    def findXSum(self, nums: List[int], k: int, x: int) -> List[int]:
        if not nums or k <= 0 or x <= 0:
            return []
        
        n = len(nums)
        answer = []
        
        # Initialize counter for the first window
        counter = Counter(nums[:k])
        
        # Function to compute x-sum given the counter and x
        def compute_x_sum(counter, x):
            # Sort elements by frequency descending, then by value descending
            sorted_elements = sorted(counter.items(), key=lambda item: (-item[1], -item[0]))
            # Select top x elements
            top_x = sorted_elements[:x]
            # Sum up all occurrences of these top x elements
            sum_x = sum(count * num for num, count in top_x)
            return sum_x
        
        # Compute x-sum for the first window
        sum_x = compute_x_sum(counter, x)
        answer.append(sum_x)
        
        # Slide the window from left to right
        for i in range(1, n - k + 1):
            # Remove the element sliding out
            counter[nums[i - 1]] -= 1
            if counter[nums[i - 1]] == 0:
                del counter[nums[i - 1]]
            # Add the new element sliding in
            counter[nums[i + k - 1]] += 1
            # Compute x-sum for the current window
            sum_x = compute_x_sum(counter, x)
            answer.append(sum_x)
        
        return answer