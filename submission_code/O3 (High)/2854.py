from typing import List

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # If there is only one word, it is already the final string
        if len(words) == 1:
            return len(words[0])

        ALPHA = 26                     # number of lower-case letters
        STATES = ALPHA * ALPHA        # (first_char, last_char) possibilities
        INF = 10 ** 9                 # a value larger than any possible answer

        # dp[first*26 + last] -> minimal length obtained so far
        dp = [INF] * STATES

        # initialise with the first word
        first = ord(words[0][0]) - 97
        last  = ord(words[0][-1]) - 97
        dp[first * 26 + last] = len(words[0])

        # process the remaining words one by one
        for w in words[1:]:
            lw = len(w)
            fw = ord(w[0])  - 97
            lwc = ord(w[-1]) - 97    # last char of the current word

            new_dp = [INF] * STATES

            for idx in range(STATES):
                cur_len = dp[idx]
                if cur_len == INF:
                    continue          # this state is unreachable at this point

                fp = idx // 26        # first char of current string
                lp = idx % 26         # last  char of current string

                # 1) append the new word on the right  ==> current + w
                add_len = cur_len + lw - (1 if lp == fw else 0)
                state  = fp * 26 + lwc
                if add_len < new_dp[state]:
                    new_dp[state] = add_len

                # 2) prepend the new word on the left  ==> w + current
                add_len = cur_len + lw - (1 if lwc == fp else 0)
                state  = fw * 26 + lp
                if add_len < new_dp[state]:
                    new_dp[state] = add_len

            dp = new_dp              # move to the next step

        return min(dp)               # best length among all possible (first, last)