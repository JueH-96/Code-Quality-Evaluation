class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Group indices by their values
        indices = defaultdict(list)
        for i, num in enumerate(nums):
            indices[num].append(i)
        
        max_length = 0
        
        # For each unique value, find the longest subarray we can make
        for value, idx_list in indices.items():
            # Use sliding window on the indices
            left = 0
            
            for right in range(len(idx_list)):
                # Current window spans from idx_list[left] to idx_list[right]
                window_start = idx_list[left]
                window_end = idx_list[right]
                window_length = window_end - window_start + 1
                count_of_value = right - left + 1
                deletions_needed = window_length - count_of_value
                
                # If we need too many deletions, shrink window from left
                while deletions_needed > k:
                    left += 1
                    window_start = idx_list[left]
                    window_length = window_end - window_start + 1
                    count_of_value = right - left + 1
                    deletions_needed = window_length - count_of_value
                
                # Update max_length with the count of this value in current window
                max_length = max(max_length, count_of_value)
        
        return max_length