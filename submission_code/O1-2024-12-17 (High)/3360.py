class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        from bisect import bisect_left, bisect_right
        
        # Count the frequency of each character.
        freq = [0]*26
        for c in word:
            freq[ord(c) - ord('a')] += 1
        
        # Collect only the positive frequencies.
        freq_list = [f for f in freq if f > 0]
        
        # If no characters, nothing to delete.
        if not freq_list:
            return 0
        
        freq_list.sort()
        n = len(freq_list)
        
        # If the existing frequencies already satisfy the k condition,
        # no deletions are needed.
        if freq_list[-1] - freq_list[0] <= k:
            return 0
        
        # Prefix sums to quickly compute subarray sums.
        prefix_sum = [0]*(n+1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + freq_list[i]
        
        total = prefix_sum[n]  # total characters in the string
        max_freq = freq_list[-1]
        
        best_keep = 0
        
        # Try each possible minimum positive frequency X from 1..max_freq
        # and clamp all frequencies to the range [X, X+k].
        for X in range(1, max_freq + 1):
            # Find where frequencies become at least X,
            # and where they exceed X+k.
            i = bisect_left(freq_list, X)
            j = bisect_right(freq_list, X + k)
            
            # Sum of frequencies that lie between [X, X+k] (unchanged).
            sum_in_range = prefix_sum[j] - prefix_sum[i]
            # The characters from j..n are all clamped to (X+k).
            clamp_count = n - j
            
            kept = sum_in_range + clamp_count * (X + k)
            best_keep = max(best_keep, kept)
        
        # Minimum deletions = total - the best number of chars we can keep
        return total - best_keep