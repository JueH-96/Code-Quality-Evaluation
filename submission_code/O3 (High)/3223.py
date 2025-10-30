class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        # Split the string into maximal segments in which every pair
        # of adjacent characters differs by at most 2.
        n = len(word)
        segments = []
        start = 0
        for i in range(1, n):
            if abs(ord(word[i]) - ord(word[i - 1])) > 2:
                segments.append(word[start:i])
                start = i
        segments.append(word[start:])

        ans = 0
        for seg in segments:
            ans += self._count_inside_segment(seg, k)
        return ans

    # ------------------------------------------------------------------
    # Count complete substrings that are fully contained in `segment`.
    # In the segment the “adjacent–difference ≤ 2” condition is already
    # satisfied, so we only have to check the “exactly k occurrences”
    # condition.
    # ------------------------------------------------------------------
    def _count_inside_segment(self, segment: str, k: int) -> int:
        length = len(segment)
        if length < k:                     # impossible to have k copies
            return 0

        res = 0
        # Try every possible number of distinct characters (1 … 26).
        for distinct in range(1, 27):
            window = distinct * k
            if window > length:            # larger windows will be impossible, break early
                break

            cnt = [0] * 26                 # frequency of every letter in current window
            unique = 0                     # how many letters have freq > 0
            satisfied = 0                  # how many letters have freq == k
            left = 0

            for right, ch in enumerate(segment):
                idx = ord(ch) - 97
                # add new character
                cnt[idx] += 1
                if cnt[idx] == 1:
                    unique += 1
                if cnt[idx] == k:
                    satisfied += 1
                elif cnt[idx] == k + 1:
                    satisfied -= 1         # went from exactly k to k+1

                # shrink if window too large
                if right - left + 1 > window:
                    idx_left = ord(segment[left]) - 97
                    before = cnt[idx_left]
                    if before == 1:
                        unique -= 1        # will become 0
                    if before == k:
                        satisfied -= 1     # will become k-1
                    elif before == k + 1:
                        satisfied += 1     # will become exactly k
                    cnt[idx_left] -= 1
                    left += 1

                # window has correct size -> check conditions
                if right - left + 1 == window and unique == distinct and satisfied == distinct:
                    res += 1
        return res