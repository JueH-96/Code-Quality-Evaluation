class Solution:
    def numberOfSubstrings(self, s: str, k: int) -> int:
        """
        We want the count of all substrings where at least one character
        appears at least k times. Equivalently, we can count how many
        substrings do NOT have any character appearing k times (i.e., the
        maximum frequency in that substring is < k) and subtract from the
        total number of substrings.

        Let bad_count = number of substrings whose max character freq < k.
        Then the answer = total_substrings - bad_count,
        where total_substrings = n*(n+1)//2 for string length n.
        """
        n = len(s)

        # Helper function to count how many substrings have maxfreq < k
        def count_substrings_with_maxfreq_less_than_k(s, k):
            freq = [0] * 26
            start = 0
            total = 0
            maxfreq = 0  # track maximum frequency in the current window

            for end in range(n):
                # Increase frequency for s[end]
                idx_end = ord(s[end]) - ord('a')
                freq[idx_end] += 1
                # Possibly update maxfreq
                maxfreq = max(maxfreq, freq[idx_end])

                # If we exceed (or reach) k, shrink from the left
                while maxfreq >= k:
                    idx_start = ord(s[start]) - ord('a')
                    freq[idx_start] -= 1
                    # If we removed a character that might have been
                    # the max frequency, recompute maxfreq
                    if freq[idx_start] + 1 == maxfreq:
                        maxfreq = max(freq)
                    start += 1

                # All substrings ending at 'end' and starting from
                # [start..end] are valid (have maxfreq < k)
                total += (end - start + 1)

            return total

        # Count substrings that do NOT have any character frequency >= k
        bad_count = count_substrings_with_maxfreq_less_than_k(s, k)

        # Total number of substrings
        total_substrings = n * (n + 1) // 2

        # Result is those that do have a character repeated >= k times
        return total_substrings - bad_count