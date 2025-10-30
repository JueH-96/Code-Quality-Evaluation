from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Convert list to set for O(1) lookups
        forbid_set = set(forbidden)
        
        n = len(word)
        # max length among forbidden strings (problem constraint: up to 10)
        max_forbidden_len = max(len(s) for s in forbidden)
        
        # r will represent the right bound (exclusive) for a valid substring starting at i.
        # Initially, r = n means that if we start at the last index, we have the whole substring as valid.
        r = n
        
        result = 0
        # Traverse from right to left. 
        # For each starting index i, determine the largest substring starting at i that does not contain any forbidden substring.
        # We only need to check substrings of length up to max_forbidden_len, because any forbidden must have length <= max_forbidden_len.
        for i in range(n - 1, -1, -1):
            # Check every potential substring starting at i, but only up to the length max_forbidden_len or until r is reached.
            # If a forbidden substring is encountered ending at position j, then we need to update r so that
            # the valid substring cannot extend to j+1.
            for j in range(i, min(n, i + max_forbidden_len)):
                # If current candidate substring is longer than our current valid range, break early.
                if j >= r:
                    break
                # Check if the substring word[i:j+1] is forbidden
                if word[i:j+1] in forbid_set:
                    # Set new r: we cannot extend past j, so r becomes min(r, j)
                    r = j
                    break  # no need to check longer substrings from i as they will include the forbidden substring as well
            # Update result. The valid substring starting at i is word[i:r]
            result = max(result, r - i)
        
        return result