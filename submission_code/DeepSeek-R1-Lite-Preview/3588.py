class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # Define mappings for creatures
        creature_map = {'F': 0, 'W': 1, 'E': 2}
        # Convert s to indices for easier access
        A = [creature_map[c] for c in s]
        # Initialize DP
        from collections import defaultdict
        # prev_dp[last][d] = number of ways
        prev_dp = defaultdict(lambda: defaultdict(int))
        # Initial state: no previous choice, difference 0
        prev_dp[-1][0] = 1  # -1 represents no previous choice
        # Define point difference based on choices
        # point_diff[a][b] = point_B - point_A
        point_diff = [[0]*3 for _ in range(3)]
        for a in range(3):
            for b in range(3):
                if a == b:
                    point_diff[a][b] = 0
                elif (a, b) in {(0,2),(1,0),(2,1)}:
                    point_diff[a][b] = -1  # Alice gets point
                else:
                    point_diff[a][b] = 1   # Bob gets point
        # Iterate through each round
        for i in range(n):
            curr_dp = defaultdict(lambda: defaultdict(int))
            for last in prev_dp:
                for d in prev_dp[last]:
                    count = prev_dp[last][d]
                    for b in range(3):
                        if b == last:
                            continue  # Bob cannot choose the same as previous
                        # Determine point difference for this round
                        pd = point_diff[A[i]][b]
                        new_d = d + pd
                        # Update the current DP
                        curr_dp[b][new_d] = (curr_dp[b][new_d] + count) % MOD
            prev_dp = curr_dp
        # Now, sum over all sequences where d > 0
        total = 0
        for last in prev_dp:
            for d in prev_dp[last]:
                if d > 0:
                    total = (total + prev_dp[last][d]) % MOD
        return total