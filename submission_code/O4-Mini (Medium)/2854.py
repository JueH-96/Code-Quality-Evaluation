from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # DP over (first_char, last_char) states.
        # dp[f][l] = minimum length of current string with first=f, last=l
        n = len(words)
        if n == 0:
            return 0
        # helper to map char -> 0..25
        def ci(c: str) -> int:
            return ord(c) - ord('a')
        INF = 10**18
        # Initialize dp for the first word
        f0 = ci(words[0][0])
        l0 = ci(words[0][-1])
        dp_prev = [[INF] * 26 for _ in range(26)]
        dp_prev[f0][l0] = len(words[0])
        # Process subsequent words
        for i in range(1, n):
            w = words[i]
            m = len(w)
            fi = ci(w[0])
            li = ci(w[-1])
            dp_cur = [[INF] * 26 for _ in range(26)]
            for f in range(26):
                for l in range(26):
                    prev_len = dp_prev[f][l]
                    if prev_len == INF:
                        continue
                    # Option 1: append w after current string
                    # overlap if last_char_of_current == first_char_of_w
                    overlap = 1 if l == fi else 0
                    new_len = prev_len + m - overlap
                    # new first = f, new last = li
                    if new_len < dp_cur[f][li]:
                        dp_cur[f][li] = new_len
                    # Option 2: prepend w before current string
                    # overlap if last_char_of_w == first_char_of_current
                    overlap = 1 if li == f else 0
                    new_len = prev_len + m - overlap
                    # new first = fi, new last = l
                    if new_len < dp_cur[fi][l]:
                        dp_cur[fi][l] = new_len
            dp_prev = dp_cur
        # The answer is the minimal length over all end states
        ans = INF
        for f in range(26):
            for l in range(26):
                if dp_prev[f][l] < ans:
                    ans = dp_prev[f][l]
        return ans