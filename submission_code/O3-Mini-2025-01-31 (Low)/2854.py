from typing import List
import math

class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # We'll use dynamic programming, where our state is represented by a tuple (first, last)
        # representing the first and last characters of the current concatenated string.
        # dp[first][last] will store the minimum length achievable with that pair.
        # Initially, we only have one string: words[0].
        
        # We'll use a 2D DP dictionary with keys being (first, last) pair.
        dp = {}
        init_first = words[0][0]
        init_last = words[0][-1]
        dp[(init_first, init_last)] = len(words[0])
        
        # Process the rest words one by one.
        for i in range(1, len(words)):
            w = words[i]
            new_dp = {}
            # For each state from previous step
            for (first, last), curr_len in dp.items():
                # Option 1: join(current, w) i.e. current string then word w
                # The deletion rule: if last character of current equals the first character of w,
                # then we remove one occurrence (the duplicate) - effectively saving 1 character.
                cost1 = len(w) - (1 if last == w[0] else 0)
                new_state_1 = (first, w[-1])
                new_len1 = curr_len + cost1
                if new_state_1 not in new_dp or new_len1 < new_dp[new_state_1]:
                    new_dp[new_state_1] = new_len1
                
                # Option 2: join(w, current) i.e. word w then current string
                # Now the deletion: if last character of w equals the first character of current string.
                cost2 = len(w) - (1 if w[-1] == first else 0)
                new_state_2 = (w[0], last)
                new_len2 = curr_len + cost2
                if new_state_2 not in new_dp or new_len2 < new_dp[new_state_2]:
                    new_dp[new_state_2] = new_len2
            dp = new_dp
        
        # The answer is the minimum length among all states.
        return min(dp.values())
        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimizeConcatenatedLength(["aa","ab","bc"]))  # Expected output: 4
    print(sol.minimizeConcatenatedLength(["ab","b"]))          # Expected output: 2
    print(sol.minimizeConcatenatedLength(["aaa","c","aba"]))   # Expected output: 6