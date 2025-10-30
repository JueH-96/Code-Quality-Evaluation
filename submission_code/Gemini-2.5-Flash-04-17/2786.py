class Solution:
    def longestSemiRepetitiveSubstring(self, s: str) -> int:
        n = len(s)
        
        # Use a list to store the indices k where s[k] == s[k+1].
        # These are the starting indices of the consecutive pairs within the current window [i, j].
        # Using a list is acceptable given the small constraint n <= 50.
        pair_indices = [] 
        
        i = 0 # Left pointer of the sliding window
        max_length = 0
        
        for j in range(n): # Right pointer of the sliding window
            # Check if adding s[j] creates a new consecutive pair s[j-1]s[j].
            # This pair starts at index j-1.
            # Add the starting index j-1 to our list of pair indices.
            if j > 0 and s[j-1] == s[j]:
                pair_indices.append(j-1)
            
            # The current window is s[i:j+1].
            # The consecutive pairs within this window are those whose start index k is
            # such that i <= k < j and s[k] == s[k+1].
            # The list `pair_indices` stores the indices k where s[k] == s[k+1] encountered
            # so far as j increased. These indices are always increasing.
            # When len(pair_indices) becomes > 1, it means the current window s[i:j+1]
            # contains more than one consecutive pair, making it not semi-repetitive.
            # We need to shrink the window from the left by increasing `i`.
            # To make the window semi-repetitive again (at most 1 pair), we must remove
            # the earliest consecutive pair from the window. The earliest pair is the one
            # whose starting index is smallest among those currently in `pair_indices`.
            # Since `pair_indices` stores indices in increasing order, the earliest pair
            # starts at index `pair_indices[0]`.
            # The new left boundary `i` must be strictly greater than `pair_indices[0]`.
            # The smallest such new `i` is `pair_indices[0] + 1`.
            while len(pair_indices) > 1:
                # Update the left window boundary `i` to be one position after the start
                # of the earliest consecutive pair (pair_indices[0]).
                # Remove the corresponding index from `pair_indices`.
                # We use pop(0) which removes the first element (earliest pair index).
                i = pair_indices.pop(0) + 1 
            
            # After potentially shrinking the window, the substring s[i:j+1] is semi-repetitive
            # because it contains at most one consecutive pair (len(pair_indices) <= 1).
            # The length of the current semi-repetitive window is j - i + 1.
            # Update the maximum length found so far.
            current_length = j - i + 1
            max_length = max(max_length, current_length)
        
        return max_length