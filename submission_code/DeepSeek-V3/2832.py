from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Dictionary to store the indices of each number
        num_indices = defaultdict(list)
        for idx, num in enumerate(nums):
            num_indices[num].append(idx)
        
        max_length = 0
        
        # Iterate through each number's indices
        for num in num_indices:
            indices = num_indices[num]
            left = 0
            # Iterate through the indices with a sliding window
            for right in range(len(indices)):
                # Calculate the number of deletions needed
                # The window is from indices[left] to indices[right]
                # The number of elements in the window is (right - left + 1)
                # The number of elements to delete is (indices[right] - indices[left] + 1) - (right - left + 1)
                # Which simplifies to (indices[right] - indices[left] + 1) - (right - left + 1) = (indices[right] - indices[left]) - (right - left)
                # Or more simply, (indices[right] - indices[left] - (right - left))
                deletions = (indices[right] - indices[left]) - (right - left)
                # If deletions exceed k, move the left pointer
                while deletions > k:
                    left += 1
                    deletions = (indices[right] - indices[left]) - (right - left)
                # Update the maximum length
                current_length = right - left + 1
                if current_length > max_length:
                    max_length = current_length
        
        return max_length