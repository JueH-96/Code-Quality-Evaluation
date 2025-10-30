from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Convert the list of forbidden strings into a hash set for O(1) average time lookups.
        forbidden_set = set(forbidden)
        
        # Determine the maximum length among all forbidden strings.
        # This optimizes the inner loop, as we only need to check substrings up to this length.
        max_f_len = 0
        if forbidden: # Handle the case where the forbidden list might be empty
            max_f_len = max(len(f) for f in forbidden)
        
        n = len(word)
        max_len = 0
        
        # `current_left_boundary` tracks the starting index of the current valid window.
        # The window being considered is `word[current_left_boundary : right + 1]`.
        current_left_boundary = 0 

        # Iterate `right` from 0 to n-1, effectively expanding the sliding window to the right.
        for right in range(n):
            # For each `right` position, we need to check if adding `word[right]`
            # forms any forbidden substring that *ends* at `right`.
            # These potential forbidden substrings can have lengths from 1 up to `max_f_len`.
            # Their starting indices would be `right - k + 1`, where `k` is the length.
            for k in range(1, max_f_len + 1): 
                start_idx = right - k + 1
                
                # If `start_idx` is negative, it means the substring would start
                # before the beginning of the `word`. There's no need to check
                # further lengths for this `right` position, so we break.
                if start_idx < 0:
                    break 
                
                # Extract the substring to check.
                sub_to_check = word[start_idx : right + 1]
                
                # If this substring is present in our `forbidden_set`:
                if sub_to_check in forbidden_set:
                    # This means `word[start_idx : right + 1]` is a forbidden string.
                    # To ensure the current window `word[current_left_boundary : right + 1]`
                    # remains valid (i.e., does not contain this forbidden substring),
                    # its `current_left_boundary` must be moved past `start_idx`.
                    # We take `max(current_left_boundary, start_idx + 1)` because
                    # `current_left_boundary` might have already been moved further right
                    # due to a previously found forbidden substring.
                    current_left_boundary = max(current_left_boundary, start_idx + 1)
            
            # After checking all relevant forbidden substrings ending at `right`
            # and potentially adjusting `current_left_boundary`,
            # the window `word[current_left_boundary : right + 1]` is guaranteed to be valid.
            # Calculate its length: `(right + 1) - current_left_boundary`.
            # Update `max_len` with the maximum valid length found so far.
            # If `current_left_boundary` somehow exceeds `right` (e.g., if even a single character is forbidden),
            # the calculated length will be <= 0, which `max` correctly handles.
            max_len = max(max_len, right - current_left_boundary + 1)
        
        return max_len