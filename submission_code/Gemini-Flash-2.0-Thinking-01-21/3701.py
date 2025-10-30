import sys

# Use a large value for infinity
INF = float('inf')

class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)

        # Minimum possible length for a good caption is 3.
        # If n < 3, it's impossible to form a good caption.
        if n < 3:
             return ""

        # dp[i][c][l]: min cost to form prefix res[0...i]
        # where res[i] is char('a' + c) and it's the (l+1)-th consecutive occurrence.
        # l = 0: length 1
        # l = 1: length 2
        # l = 2: length >= 3
        dp = [[[INF for _ in range(3)] for _ in range(26)] for _ in range(n)]
        parent = [[ [(-1, -1) for _ in range(3)] for _ in range(26)] for _ in range(n)]

        # Base case i = 0
        for c in range(26):
            dp[0][c][0] = abs(ord('a' + c) - ord(caption[0]))

        # Fill DP table
        for i in range(1, n):
            for c in range(26):
                cost_i = abs(ord('a' + c) - ord(caption[i]))

                # Calculate dp[i][c][0] (length 1 at i): res[i]=c, res[i-1] != c
                min_prev_cost_l0 = INF
                best_prev_c_l0 = -1

                if i == 1:
                    # From dp[0][prev_c][0], prev_c != c
                    for prev_c in range(26):
                        if prev_c != c:
                           if dp[0][prev_c][0] < min_prev_cost_l0:
                             min_prev_cost_l0 = dp[0][prev_c][0]
                             best_prev_c_l0 = prev_c
                           # Tie-breaking: Prefer smaller prev_c when costs are equal
                           elif dp[0][prev_c][0] == min_prev_cost_l0 and best_prev_c_l0 != -1 and prev_c < best_prev_c_l0:
                              best_prev_c_l0 = prev_c
                    if min_prev_cost_l0 != INF:
                        dp[i][c][0] = cost_i + min_prev_cost_l0
                        parent[i][c][0] = (best_prev_c_l0, 0) # length 0

                elif i >= 2:
                    # From dp[i-1][prev_c][2], prev_c != c. Requires dp[i-1][prev_c][2] to be finite.
                    for prev_c in range(26):
                        if prev_c != c:
                            if dp[i-1][prev_c][2] < min_prev_cost_l0:
                                min_prev_cost_l0 = dp[i-1][prev_c][2]
                                best_prev_c_l0 = prev_c
                            # Tie-breaking: Prefer smaller prev_c when costs are equal
                            elif dp[i-1][prev_c][2] == min_prev_cost_l0 and best_prev_c_l0 != -1 and prev_c < best_prev_c_l0:
                                best_prev_c_l0 = prev_c
                    if min_prev_cost_l0 != INF:
                        dp[i][c][0] = cost_i + min_prev_cost_l0
                        parent[i][c][0] = (best_prev_c_l0, 2) # length 2 corresponds to >= 3


                # Calculate dp[i][c][1] (length 2): res[i]=c, res[i-1]=c, res[i-2] != c
                # Requires res[i-1]=c was length 1 (l=0) at i-1
                # Transition is valid only if i >= 1.
                if i >= 1 and dp[i-1][c][0] != INF:
                    dp[i][c][1] = cost_i + dp[i-1][c][0]
                    parent[i][c][1] = (c, 0)

                # Calculate dp[i][c][2] (length >= 3): res[i]=c, res[i-1]=c, res[i-2]=c
                # Requires res[i-1]=c was length >= 2 (l=1 or l=2) at i-1
                # Transition is valid only if i >= 2.
                if i >= 2:
                    cost1 = dp[i-1][c][1]
                    cost2 = dp[i-1][c][2]
                    min_prev_l_cost = INF
                    best_prev_l = -1

                    if cost1 < cost2:
                        min_prev_l_cost = cost1
                        best_prev_l = 1
                    elif cost2 < cost1:
                        min_prev_l_cost = cost2
                        best_prev_l = 2
                    elif cost1 != INF: # cost1 == cost2 != INF
                        min_prev_l_cost = cost1
                        # Tie-breaking for equal costs from l=1 vs l=2 at i-1
                        # If c > 0 ('a'), prefer prev_l=1 (allows != c at i-2).
                        # If c == 0 ('a'), prefer prev_l=2 (forces == c at i-2).
                        if c > 0:
                           best_prev_l = 1
                        else: # c == 0
                           best_prev_l = 2

                    if min_prev_l_cost != INF:
                        dp[i][c][2] = cost_i + min_prev_l_cost
                        parent[i][c][2] = (c, best_prev_l)

        # Find the minimum cost among valid final states (length >= 3 at n-1)
        min_cost = INF
        best_last_c = -1

        for c in range(26):
            # The final state must have length >= 3 (l=2)
            if dp[n-1][c][2] < min_cost:
                min_cost = dp[n-1][c][2]
                best_last_c = c
            # Tie-breaking: Prefer smaller character at the end if costs are tied
            elif dp[n-1][c][2] == min_cost:
                 # If best_last_c hasn't been set yet (first valid candidate), or current c is smaller
                 if best_last_c == -1 or c < best_last_c:
                    best_last_c = c


        # If no good caption found
        if best_last_c == -1 or min_cost == INF:
            return ""

        # Backtrack to reconstruct the string
        result = []
        curr_i = n - 1
        curr_c = best_last_c
        curr_l = 2 # Final state must be length >= 3

        while curr_i >= 0:
            result.append(chr(ord('a') + curr_c))
            if curr_i == 0:
                break

            # Get parent state (prev_c, prev_l)
            prev_c, prev_l = parent[curr_i][curr_c][curr_l]

            # This check is redundant if min_cost was finite, but included for robustness
            # (means parent was not set, which shouldn't happen if DP value is finite)
            if prev_c == -1:
                 # Should not happen if min_cost was finite. Indicates an issue.
                 # Returning "" implies impossibility, consistent with spec.
                 return ""


            curr_i -= 1
            curr_c = prev_c
            curr_l = prev_l

        # The result was built backward, reverse it
        return "".join(reversed(result))