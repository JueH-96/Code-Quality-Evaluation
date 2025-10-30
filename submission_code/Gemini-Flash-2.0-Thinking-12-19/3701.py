class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        # caption length < 3, impossible to form groups of size >= 3
        if n < 3:
            return ""

        # dp[i][c_code][length_state]
        # length_state 0: length 1
        # length_state 1: length 2
        # length_state 2: length >= 3
        # c_code: 0 for 'a', ..., 25 for 'z'
        inf = float('inf')
        dp = [[[inf] * 3 for _ in range(26)] for _ in range(n)]

        # Base case i = 0
        for c_code in range(26):
            dp[0][c_code][0] = abs(ord(caption[0]) - ord('a') - c_code)

        # Fill DP table
        for i in range(1, n):
            for c_code in range(26):
                cost = abs(ord(caption[i]) - ord('a') - c_code)

                # Transition to state (i, c_code, 0): start of new block (length 1)
                # Previous block ending at i-1 must be good (length >= 3)
                min_prev_cost_good_block = inf
                for prev_c_code in range(26):
                    if prev_c_code != c_code:
                        min_prev_cost_good_block = min(min_prev_cost_good_block, dp[i-1][prev_c_code][2])
                
                if min_prev_cost_good_block != inf:
                     dp[i][c_code][0] = cost + min_prev_cost_good_block

                # Transition to state (i, c_code, 1): second char in block (length 2)
                # Previous state must be (i-1, c_code, 0)
                if dp[i-1][c_code][0] != inf:
                    dp[i][c_code][1] = cost + dp[i-1][c_code][0]

                # Transition to state (i, c_code, 2): third or later char in block (length >= 3)
                # Previous state must be (i-1, c_code, 1) or (i-1, c_code, 2)
                min_prev_cost_extend_block = min(dp[i-1][c_code][1], dp[i-1][c_code][2])
                
                if min_prev_cost_extend_block != inf:
                    dp[i][c_code][2] = cost + min_prev_cost_extend_block

        # Find minimum cost at the end (index n-1), ending in a good block (length >= 3)
        min_cost = inf
        for c_code in range(26):
            min_cost = min(min_cost, dp[n-1][c_code][2])

        # If minimum cost is still infinity, no solution exists
        if min_cost == inf:
            return ""

        # Reconstruct the lexicographically smallest string
        result = []
        curr_i = n - 1
        curr_c_code = -1
        curr_length_state = 2 # Must end in a good block

        # Find the starting character (smallest c_code) for the minimum cost ending state
        for c_code in range(26):
            if dp[curr_i][c_code][curr_length_state] == min_cost:
                curr_c_code = c_code
                break # Found the lexicographically smallest starting character

        # Backtrack
        while curr_i >= 0:
            result.append(chr(curr_c_code + ord('a')))

            if curr_i == 0:
                break # Finished

            cost = abs(ord(caption[curr_i]) - ord('a') - curr_c_code)
            required_prev_cost = dp[curr_i][curr_c_code][curr_length_state] - cost

            if curr_length_state == 0: # Current state (i, c, 0)
                # Previous state must be (i-1, prev_c, 2) where prev_c != c
                # Find the lexicographically smallest prev_c_code that matches the required_prev_cost
                found_prev = False
                for prev_c_code in range(26): # Iterate prev_c_code from smallest to largest
                    if prev_c_code != curr_c_code and dp[curr_i-1][prev_c_code][2] == required_prev_cost:
                         curr_c_code = prev_c_code
                         curr_length_state = 2
                         found_prev = True
                         break # Found the lexicographically smallest previous char
                # assert found_prev # Should always find a valid previous state if DP value is finite

            elif curr_length_state == 1: # Current state (i, c, 1)
                # Previous state must be (i-1, c, 0)
                # assert dp[curr_i-1][curr_c_code][0] == required_prev_cost # Should always match
                curr_length_state = 0

            elif curr_length_state == 2: # Current state (i, c, 2)
                # Previous state could be (i-1, c, 1) or (i-1, c, 2)
                # Prefer k=1 for lexicographical order if its DP value matches the required_prev_cost
                
                match_k1 = (dp[curr_i-1][curr_c_code][1] == required_prev_cost)
                # match_k2 = (dp[curr_i-1][curr_c_code][2] == required_prev_cost) # This must match if match_k1 doesn't and dp[curr_i][curr_c_code][2] is finite

                if match_k1: # If state (i-1, c, 1) is a valid predecessor by cost
                    curr_length_state = 1
                else: # State (i-1, c, 1) is not a valid predecessor by cost, state (i-1, c, 2) must be
                     # assert dp[curr_i-1][curr_c_code][2] == required_prev_cost
                     curr_length_state = 2

            curr_i -= 1

        result.reverse() # Reverse the list to get the correct order
        return "".join(result)