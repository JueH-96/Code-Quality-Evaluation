class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Group positions by value
        positions = defaultdict(list)
        for i, num in enumerate(nums):
            positions[num].append(i)
        
        max_length = 0
        
        for pos_list in positions.values():
            # Use sliding window to find the maximum number of occurrences
            # we can keep such that we delete at most k elements
            left = 0
            for right in range(len(pos_list)):
                # Ensure the number of deletions is at most k
                while pos_list[right] - pos_list[left] - (right - left) > k:
                    left += 1
                
                # Update the maximum length
                max_length = max(max_length, right - left + 1)
        
        return max_length