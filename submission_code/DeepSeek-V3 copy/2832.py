from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Dictionary to store the indices of each number
        num_indices = defaultdict(list)
        for idx, num in enumerate(nums):
            num_indices[num].append(idx)
        
        max_length = 0
        
        # Iterate through each number and its indices
        for num in num_indices:
            indices = num_indices[num]
            left = 0
            # Iterate through the indices with a sliding window
            for right in range(len(indices)):
                # Calculate the number of deletions needed
                # The window is from indices[left] to indices[right]
                # The number of elements in the window is (right - left + 1)
                # The number of elements in the original array between indices[left] and indices[right] is (indices[right] - indices[left] + 1)
                # The number of deletions is (indices[right] - indices[left] + 1) - (right - left + 1)
                deletions = (indices[right] - indices[left] + 1) - (right - left + 1)
                # If deletions exceed k, move the left pointer
                while deletions > k:
                    left += 1
                    deletions = (indices[right] - indices[left] + 1) - (right - left + 1)
                # Update the maximum length
                max_length = max(max_length, right - left + 1)
        
        return max_length