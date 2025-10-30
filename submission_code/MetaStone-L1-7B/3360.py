from collections import Counter
import bisect

class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        # Count the frequency of each character
        freq = Counter(word)
        # Extract the list of frequencies
        freq_list = list(freq.values())
        # Sort the frequencies
        freq_list.sort()
        n = len(freq_list)
        # Compute prefix sums
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i+1] = prefix_sum[i] + freq_list[i]
        sum_f = sum(freq_list)
        min_f = freq_list[0] if n > 0 else 0
        
        min_deletions = float('inf')
        
        for a in range(0, min_f + 1):
            s = a + k
            # Find the first index where freq_list[j] > s
            j = bisect.bisect_right(freq_list, s)
            sum_min = prefix_sum[j] + s * (n - j)
            deletions = sum_f - sum_min
            if deletions < min_deletions:
                min_deletions = deletions
        
        return min_deletions

# Example usage:
# sol = Solution()
# print(sol.minimumDeletions("aabcaba", 0))  # Output: 3
# print(sol.minimumDeletions("dabdcbdcb", 2))  # Output: 0
# print(sol.minimumDeletions("aaabaaa", 1))    # Output: 1