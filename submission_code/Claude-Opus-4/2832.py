class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Group indices by value
        indices = defaultdict(list)
        for i, num in enumerate(nums):
            indices[num].append(i)
        
        max_length = 0
        
        # For each unique value, find the longest subarray
        for positions in indices.values():
            if not positions:
                continue
                
            # Use sliding window on positions
            left = 0
            
            for right in range(len(positions)):
                # Calculate deletions needed between positions[left] and positions[right]
                # This is the total distance minus the number of elements of our value
                while positions[right] - positions[left] - (right - left) > k:
                    left += 1
                
                # Update max length
                max_length = max(max_length, right - left + 1)
        
        return max_length