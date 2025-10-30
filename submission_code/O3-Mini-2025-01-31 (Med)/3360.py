class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count the frequency of each character.
        freq = {}
        for ch in word:
            freq[ch] = freq.get(ch, 0) + 1
        
        # Convert frequencies to a list.
        counts = list(freq.values())
        max_freq = max(counts)
        best = float('inf')
        
        # We will choose a candidate lower bound "L" for the frequencies we keep.
        # For each letter with frequency f:
        #   - if f < L, then we cannot "raise" its frequency so we must delete all its occurrences.
        #   - if f is above the allowed upper bound (which is L+k), we can trim it down to L+k.
        #   - if f is in between L and L+k (inclusive), no deletion is needed.
        #
        # We try every candidate L from 0 up to max frequency. The allowed range will then be [L, L+k].
        for L in range(0, max_freq + 1):
            total_deletions = 0
            lower = L
            upper = L + k
            for count in counts:
                if count < lower:
                    # Frequency too low to fall in range; remove all occurrences.
                    total_deletions += count
                elif count > upper:
                    # Frequency is too high; remove the extra characters.
                    total_deletions += count - upper
                # If count is in [lower, upper], no deletion needed.
            best = min(best, total_deletions)
            
        return best

# Sample testing.
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    word1 = "aabcaba"
    k1 = 0
    print(sol.minimumDeletions(word1, k1))  # Expected output: 3

    # Example 2:
    word2 = "dabdcbdcdcd"
    k2 = 2
    print(sol.minimumDeletions(word2, k2))  # Expected output: 2

    # Example 3:
    word3 = "aaabaaa"
    k3 = 2
    print(sol.minimumDeletions(word3, k3))  # Expected output: 1