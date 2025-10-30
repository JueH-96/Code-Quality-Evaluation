class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        # Convert requirements to a dictionary for easy lookup
        req_dict = {end + 1: cnt for end, cnt in requirements}
        
        # Maximum possible inversions
        max_inversions = 401  # Based on constraints
        
        # dp[i][j] = number of permutations of length i with j inversions
        dp = [[0] * max_inversions for _ in range(n + 1)]
        dp[0][0] = 1
        
        for i in range(1, n + 1):
            # Check if there's a requirement for this prefix length
            if i in req_dict:
                # We must have exactly req_dict[i] inversions
                required_inversions = req_dict[i]
                for prev_inv in range(max_inversions):
                    if dp[i-1][prev_inv] == 0:
                        continue
                    # We can add 0 to i-1 inversions by placing the new element
                    for new_inv in range(min(i, max_inversions - prev_inv)):
                        if prev_inv + new_inv == required_inversions:
                            dp[i][required_inversions] = (dp[i][required_inversions] + dp[i-1][prev_inv]) % MOD
            else:
                # No requirement, so we can have any number of inversions
                for j in range(max_inversions):
                    if dp[i-1][j] == 0:
                        continue
                    # Place the new element in positions 0 to i-1
                    for k in range(min(i, max_inversions - j)):
                        dp[i][j + k] = (dp[i][j + k] + dp[i-1][j]) % MOD
        
        # The answer is the number of permutations of length n
        # with the required number of inversions for the full array
        if n in req_dict:
            return dp[n][req_dict[n]]
        else:
            # If no requirement for full array, sum all possibilities
            return sum(dp[n]) % MOD