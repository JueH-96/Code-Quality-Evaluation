class Solution:
    def longestEqualSubarray(self, nums: List[int], k: int) -> int:
        from collections import defaultdict
        
        # Dictionary mapping each number in nums to the list of its indices (in ascending order)
        positions = defaultdict(list)
        for i, val in enumerate(nums):
            positions[val].append(i)
        
        answer = 0

        # For each distinct value, use a sliding window on its list of positions
        # We want to find the largest window [left..right] such that
        # positions[right] - positions[left] - (right - left) <= k.
        # This cost represents how many non-this-value elements are in between
        # these positions in the original array (which we can delete at most k).
        for val, indices in positions.items():
            left = 0
            for right in range(len(indices)):
                # Shrink the window from the left if cost exceeds k
                while indices[right] - indices[left] - (right - left) > k:
                    left += 1
                # Update the best valid window size
                answer = max(answer, right - left + 1)
        
        return answer