import collections
import bisect

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        """
        Calculates the minimum number of deletions to make a word k-special.
        """
        # Calculate the frequency of each character in the input `word`.
        freq_map = collections.Counter(word)
        
        # Get a list of the non-zero frequencies.
        counts = list(freq_map.values())
        
        # If there are 0 or 1 distinct characters, the word is already k-special,
        # so no deletions are needed.
        if len(counts) <= 1:
            return 0
            
        # Sort the frequencies. This allows for efficient range-based calculations.
        counts.sort()
        
        n = len(counts)
        
        # Create a prefix sum array to quickly calculate the sum of frequencies
        # in any sub-array. `prefix_sum[i]` stores the sum of `counts[0...i-1]`.
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + counts[i]
            
        # Initialize the minimum deletions to a large value.
        min_deletions = len(word)
        
        # The core idea is that in a k-special string, all character frequencies
        # must lie within a range [min_f, min_f + k]. We iterate through each
        # original frequency, treat it as a potential `min_f` (`base_freq`), and
        # calculate the required deletions.
        
        for i in range(n):
            base_freq = counts[i]
            
            # Scenario: We choose `base_freq` as the minimum frequency of our target range.
            # The allowed frequency range is [base_freq, base_freq + k].
            
            # 1. Deletions from characters with frequencies smaller than `base_freq`.
            # These characters must be deleted entirely. Their frequencies are
            # `counts[0...i-1]`, and their sum is `prefix_sum[i]`.
            deletions_from_smaller = prefix_sum[i]
            
            # 2. Deletions from characters with frequencies larger than `base_freq + k`.
            # These frequencies must be reduced to `base_freq + k`.
            limit = base_freq + k
            
            # Find the first index `j` where `counts[j] > limit` using binary search.
            # All frequencies from `counts[j]` onwards are too large.
            j = bisect.bisect_right(counts, limit)
            
            deletions_from_larger = 0
            if j < n:
                # Sum of all frequencies that are too large (from index j to end).
                sum_of_larger_freqs = prefix_sum[n] - prefix_sum[j]
                
                # Number of such frequencies.
                num_larger_freqs = n - j
                
                # We must reduce each of these frequencies to `limit`. The number of
                # deletions is the original total sum minus what we keep.
                deletions_from_larger = sum_of_larger_freqs - (num_larger_freqs * limit)
                
            # Total deletions for the current choice of `base_freq`.
            current_deletions = deletions_from_smaller + deletions_from_larger
            
            # Update the overall minimum deletions.
            min_deletions = min(min_deletions, current_deletions)
            
        return min_deletions