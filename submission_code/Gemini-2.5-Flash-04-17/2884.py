from typing import List

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        # Convert forbidden list to a set for efficient lookup.
        forbidden_set = set(forbidden)

        # The maximum possible length of a forbidden string is 10, as per constraints.
        # We only need to check substrings of word up to this length ending at the current position.
        max_check_len = 10

        n = len(word)
        max_len = 0
        # earliest_start tracks the earliest possible start index (inclusive)
        # for the current valid substring ending at index j.
        # A valid substring word[i...j] must not contain any forbidden string.
        # If word[p...q] is a forbidden string with q <= j, then for word[i...j] to be valid,
        # we must have i > p, or i >= p + 1.
        # earliest_start will be the maximum of (p + 1) for all forbidden strings word[p...q] with q <= j.
        # Initialize to 0, meaning a substring can potentially start from index 0.
        earliest_start = 0

        # Iterate through the word string, considering each position j as the potential end of a valid substring.
        for j in range(n):
            # For the current end position j, check for any forbidden substrings ending exactly at j.
            # Check lengths k from 1 up to max_check_len.
            # A substring of length k ending at j starts at index j - k + 1.
            for k in range(1, max_check_len + 1):
                 sub_start = j - k + 1

                 # Ensure the substring starts within the bounds of word (i.e., sub_start >= 0).
                 # If sub_start < 0, it means the current length k is greater than j + 1,
                 # so a substring of length k cannot end at j.
                 if sub_start < 0:
                     # Since increasing k further will also result in sub_start < 0,
                     # we can break the inner loop for k for this j.
                     break

                 # Get the substring of length k ending at j.
                 sub = word[sub_start : j + 1]

                 # Check if this substring is in the forbidden set.
                 if sub in forbidden_set:
                     # Found a forbidden substring ending at j.
                     # Its start index is sub_start.
                     # Any valid substring ending at j must start strictly after sub_start.
                     # The earliest allowed start index considering this specific forbidden occurrence is sub_start + 1.
                     # Update earliest_start to be the maximum of its current value
                     # and this new constraint (sub_start + 1).
                     # By taking max, we ensure earliest_start respects the constraint
                     # imposed by *all* forbidden substrings ending at or before j that
                     # we have processed up to this point.
                     earliest_start = max(earliest_start, sub_start + 1)

            # After checking all relevant forbidden substrings ending at j,
            # the earliest_start variable holds the minimum valid start index
            # for a substring ending at j.
            # The longest valid substring ending at j starts at earliest_start.
            # Its length is j - earliest_start + 1.
            # We take the maximum length found so far across all possible ending positions j.
            # If earliest_start > j, the length j - earliest_start + 1 will be <= 0,
            # which correctly indicates that no valid non-empty substring ends at j
            # starting from or after earliest_start. The max() function handles this naturally,
            # ensuring max_len is non-negative (initialized to 0).
            max_len = max(max_len, j - earliest_start + 1)

        return max_len