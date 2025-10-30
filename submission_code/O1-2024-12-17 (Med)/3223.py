class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        """
        We say a substring s of 'word' is 'complete' if:
          1) Each character in s occurs exactly k times.
          2) For every pair of adjacent characters c1, c2 in s, |ord(c1) - ord(c2)| <= 2.

        We return the count of all such substrings in 'word'.
        
        --------------------------------------------------------
        EXPLANATION OF THE APPROACH:

        1) Partition the string into "segments" where each segment
           is a maximal run of indices i..j such that for every
           adjacent pair within that run, the difference in their
           character codes is <= 2. (Whenever we encounter a gap
           of > 2 between word[i] and word[i+1], we start a new segment.)

           Reason: Any valid substring cannot cross a boundary
           where adjacent characters differ by >= 3, so it must lie
           fully in one of these segments.

        2) Within each such segment, we count (using a 2-pointer/sliding-window)
           the number of substrings in which each character appears
           exactly k times.

           - Let left be the left boundary of our current window,
             and right move from the start to the end of the segment.
           - Maintain a frequency array freq of size 26 (for 'a'..'z'),
             plus:
                distinct_count = number of characters that have freq > 0
                count_k       = number of characters whose freq == k
           - We expand right by 1 each step, update freq[word[right]].
             If freq exceeds k for that character, or if the window is too large
             (window_size > distinct_count * k), we move left forward until
             those conditions are resolved.
           - Whenever our window satisfies:
                (window_length == distinct_count * k
                 AND distinct_count == count_k),
             it means each distinct character in the window has exactly k occurrences.
             We increment our answer by 1.

        3) Summation of counts across all segments is our final result.

        This approach runs in O(n) time overall, as each index
        is visited at most twice (once by 'right' and once by 'left').
        """

        #-------------------------
        # STEP 1: Partition 'word' into valid segments where
        #         adjacent-character differences are <= 2.
        #-------------------------
        segments = []
        start = 0
        for i in range(1, len(word)):
            if abs(ord(word[i]) - ord(word[i - 1])) > 2:
                # Close off the previous segment
                segments.append((start, i - 1))
                start = i
        # Last segment
        segments.append((start, len(word) - 1))

        # A helper to increment freq[c], updating distinct_count/count_k.
        def inc_char(c):
            old = freq[c]
            freq[c] += 1
            if old == 0:
                self_distinct[0] += 1  # distinct_count
            if freq[c] == k:
                self_countk[0] += 1
            if old == k:
                # it was k, now k+1
                self_countk[0] -= 1

        # A helper to decrement freq[c], updating distinct_count/count_k.
        def dec_char(c):
            old = freq[c]
            freq[c] -= 1
            if freq[c] == 0:
                self_distinct[0] -= 1
            if old == k:
                # it was k, now k-1
                self_countk[0] -= 1
            if freq[c] == k:
                self_countk[0] += 1

        answer = 0

        #-------------------------
        # STEP 2: For each segment, use a sliding window to find
        #         all substrings that meet the "exactly k times each char" condition.
        #-------------------------
        for seg_start, seg_end in segments:
            freq = [0]*26
            # We'll keep these in lists-of-one so we can mutate inside helper funcs
            self_distinct = [0]  # number of characters freq>0
            self_countk   = [0]  # number of characters freq==k

            left = seg_start
            for right in range(seg_start, seg_end + 1):
                c_right = ord(word[right]) - ord('a')
                inc_char(c_right)

                # Now shrink from the left if needed
                while True:
                    window_len = right - left + 1
                    # If the window is "too large" for the distinct_count,
                    # or if any frequency is > k (which we check indirectly by
                    # comparing freq[c_right] because that's the only one
                    # that might have exceeded k just now),
                    # we shrink from the left.
                    if window_len > self_distinct[0] * k or freq[c_right] > k:
                        c_left = ord(word[left]) - ord('a')
                        dec_char(c_left)
                        left += 1
                    else:
                        break

                # Check if the current window is a valid "complete" substring
                # Condition: each distinct char freq == k
                # => window_len = self_distinct[0] * k
                #    and self_distinct[0] == self_countk[0].
                window_len = right - left + 1
                if window_len == self_distinct[0] * k and self_distinct[0] == self_countk[0]:
                    answer += 1

        return answer