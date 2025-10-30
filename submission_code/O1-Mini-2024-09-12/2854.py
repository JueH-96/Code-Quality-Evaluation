from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        INF = float('inf')
        dp = [[INF]*26 for _ in range(26)]
        first = ord(words[0][0]) - ord('a')
        last = ord(words[0][-1]) - ord('a')
        dp[first][last] = len(words[0])

        for word in words[1:]:
            new_dp = [[INF]*26 for _ in range(26)]
            w_first = ord(word[0]) - ord('a')
            w_last = ord(word[-1]) - ord('a')
            w_len = len(word)
            for f in range(26):
                for l in range(26):
                    if dp[f][l] == INF:
                        continue
                    curr_len = dp[f][l]
                    # Append word to the current string
                    if l == w_first:
                        total_len = curr_len + w_len - 1
                    else:
                        total_len = curr_len + w_len
                    nf, nl = f, w_last
                    if total_len < new_dp[nf][nl]:
                        new_dp[nf][nl] = total_len
                    # Prepend word to the current string
                    if w_last == f:
                        total_len_prep = curr_len + w_len - 1
                    else:
                        total_len_prep = curr_len + w_len
                    nf_prep, nl_prep = w_first, l
                    if total_len_prep < new_dp[nf_prep][nl_prep]:
                        new_dp[nf_prep][nl_prep] = total_len_prep
            dp = new_dp

        min_len = INF
        for f in range(26):
            for l in range(26):
                if dp[f][l] < min_len:
                    min_len = dp[f][l]
        return min_len