import collections

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # 1. Count character frequencies
        counts = collections.Counter(word)

        # Get the list of positive frequencies
        # Problem constraints guarantee 1 <= word.length, so counts will not be empty.
        original_freqs = list(counts.values())

        # Get unique positive frequencies and sort them.
        unique_freqs = sorted(list(set(original_freqs)))

        n = len(word)
        min_deletions = n # Initialize with the maximum possible deletions (delete all characters)

        # Candidate values for the minimum frequency (L) in the target k-special string.
        # The optimal minimum frequency must be 0 or align with one of the original frequencies.
        # If the optimal range is [L_opt, R_opt] with R_opt - L_opt <= k,
        # and L_opt is not 0 or any original frequency, we can shift the range
        # to align L with the next smaller unique frequency (or 0) without
        # increasing deletions from the lower end, and potentially decreasing
        # deletions from the upper end (since R=L+k decreases).
        candidate_l_values = sorted(list(set([0] + unique_freqs)))

        # Iterate through possible values for the minimum frequency (L) in the target k-special string
        for l in candidate_l_values:
            # If L is the minimum frequency allowed, the maximum frequency (R)
            # can be at most L + k. The target range for frequencies is [L, L+k].
            r = l + k

            current_deletions = 0
            # Calculate deletions required to make frequencies fall into the range [L, R]
            for freq in original_freqs:
                if freq < l:
                    # If original frequency is less than the minimum allowed L,
                    # we must delete all occurrences of this character. Its frequency becomes 0.
                    current_deletions += freq
                elif freq > r:
                    # If original frequency is greater than the maximum allowed R,
                    # we must delete occurrences to bring it down to R.
                    current_deletions += freq - r
                # If l <= freq <= r, we keep this frequency as is. 0 deletions needed for this char
                # based on the range requirement.

            min_deletions = min(min_deletions, current_deletions)

        return min_deletions