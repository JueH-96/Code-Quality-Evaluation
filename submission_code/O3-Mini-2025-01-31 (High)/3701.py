class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        INF = 10**12  # a large number

        # if n is too short then it is impossible to form a good caption
        if n < 3:
            return ""
        
        # Precompute prefix sums for each target letter.
        # P[c][i] will store the cost to convert S[0...i-1] to letter (chr(c + ord('a'))).
        P = [[0]*(n+1) for _ in range(26)]
        for i, ch in enumerate(caption):
            orig = ord(ch) - ord('a')
            for c in range(26):
                # cost to change S[i] to letter c is |orig - c|
                P[c][i+1] = P[c][i] + abs(orig - c)
        
        # dp_cost[i] will be the minimum cost to transform caption[i:]
        # dp_str[i] will be the corresponding (lexicographically smallest) good caption from i.
        dp_cost = [INF]*(n+1)
        dp_str  = [None]*(n+1)
        # For i==n, we have an "empty" good caption with cost 0.
        dp_cost[n] = 0
        dp_str[n]  = ""
        
        # For i such that n-i < 3 (but i != n), no valid transformation is possible.
        for i in range(n-2, -1, -1):
            if n - i < 3:
                dp_cost[i] = INF
                dp_str[i]  = None

        # We'll build the dp arrays in descending order.
        # In order to “avoid looping over all j” in the dp recurrence we maintain, for each letter, an array B.
        # For each candidate letter c (0 ... 25) and for each index i, 
        # B[c][i] = (val, s, j) where:
        #   val = P[c][j] + dp_cost[j]   (i.e. the “aggregate” cost for the suffix starting at j, if we choose letter c as the block’s letter)
        #   s   = dp_str[j]             (the good caption from j)
        #   j   = the index in [i, n] at which this value achieves its minimum 
        # with the “side constraint” that if j < n then dp_str[j] must not start with letter (chr(c + 'a')).
        B = [[(0, "", 0)]*(n+1) for _ in range(26)]
        # For index n, B[c][n] = ( P[c][n] + dp_cost[n], dp_str[n], n )
        for c in range(26):
            B[c][n] = (P[c][n], "", n)
        
        # A helper function to compare two candidate tuples (val, s, j). We compare by:
        #   1. cost (val)
        #   2. then lexicographically by s.
        def min_tuple(a, b):
            # a and b are tuples: (val, s, j)
            if a[0] != b[0]:
                return a if a[0] < b[0] else b
            # if cost tie, compare by string
            if a[1] != b[1]:
                return a if a[1] < b[1] else b
            return a if a[2] <= b[2] else b
        
        # Now fill dp and B arrays backward.
        # (We must update dp[i] using the “B arrays” for index i+3.)
        for i in range(n-1, -1, -1):
            # Only consider indices where there is room for a segment of length at least 3.
            # (For i with n-i < 3 we already set dp_str[i] = None.)
            if n - i < 3:
                # (These indices remain with dp_cost = INF and dp_str = None)
                pass
            else:
                best_cost = INF
                best_str  = None
                best_choice_letter = None
                best_choice_j = None
                # Try each possible letter (from 'a' to 'z') for the block starting at i.
                # (We will “query” our B–array for that letter at index i+3.)
                for c in range(26):
                    c_char = chr(c + ord('a'))
                    j0 = i + 3
                    if j0 > n:
                        continue
                    # B[c][j0] is computed already (since j0 > i).
                    val, s_suffix, j_candidate = B[c][j0]
                    # candidate total cost = (P[c][j_candidate] + dp_cost[j_candidate]) – P[c][i].
                    candidate_cost = val - P[c][i]
                    if candidate_cost >= INF:
                        continue  # not a valid candidate
                    # The block we form is from i to j_candidate-1 (all turned into letter c_char)
                    # and then the good caption from j_candidate.
                    candidate_block = c_char * (j_candidate - i)
                    candidate_str = candidate_block + s_suffix
                    # update best candidate (compare first cost then lex)
                    if candidate_cost < best_cost or (candidate_cost == best_cost and (best_str is None or candidate_str < best_str)):
                        best_cost = candidate_cost
                        best_str  = candidate_str
                        best_choice_letter = c
                        best_choice_j = j_candidate
                dp_cost[i] = best_cost
                dp_str[i]  = best_str
                # (We could also store dp_choice[i] = (best_choice_letter, best_choice_j)
                #  if we wished to reconstruct without storing dp_str[i].)
            
            # Now update the B arrays at index i.
            # For each target letter, we “offer” the candidate coming from dp[i] IF it is allowed.
            # The candidate is allowed only if dp_str[i] exists and (if i < n) its first character is not equal
            # to that letter – this is our “side–constraint.”
            for c in range(26):
                c_char = chr(c + ord('a'))
                if dp_str[i] is not None and dp_str[i] != "" and dp_str[i][0] == c_char:
                    candidate_i = (INF, "", i)
                else:
                    candidate_i = (P[c][i] + dp_cost[i], dp_str[i] if dp_str[i] is not None else "", i)
                # When i == n this update is not needed because we already initialized B[c][n].
                if i == n:
                    B[c][i] = candidate_i
                else:
                    # B[c][i] = min( candidate_i, B[c][i+1] )
                    # (We use our helper min_tuple function to pick the better candidate by cost then by lex.)
                    B[c][i] = min_tuple(candidate_i, B[c][i+1])
        
        # The answer is dp_str[0] if dp_cost[0] is not INF.
        return dp_str[0] if dp_cost[0] < INF else ""
        

# For testing (you can remove or comment out the following block before submitting)
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    caption = "cdcd"
    print("Input:", caption)
    print("Output:", sol.minCostGoodCaption(caption))  # Expected "cccc"

    # Example 2:
    caption = "aca"
    print("Input:", caption)
    print("Output:", sol.minCostGoodCaption(caption))  # Expected "aaa"

    # Example 3:
    caption = "bc"
    print("Input:", caption)
    print("Output:", sol.minCostGoodCaption(caption))  # Expected ""