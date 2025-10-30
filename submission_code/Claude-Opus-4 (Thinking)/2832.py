from collections import defaultdict

class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        # Group indices by their values
        indices = defaultdict(list)
        for i, num in enumerate(nums):
            indices[num].append(i)
        
        max_length = 0
        
        # For each unique value
        for positions in indices.values():
            # Use sliding window on positions
            left = 0
            for right in range(len(positions)):
                # Calculate deletions needed between positions[left] and positions[right]
                while positions[right] - positions[left] - (right - left) > k:
                    left += 1
                
                # Update max_length
                max_length = max(max_length, right - left + 1)
        
        return max_length