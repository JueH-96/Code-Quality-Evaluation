MOD = 10**9 + 7

class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        # Precompute the points Alice gets for each round
        alice_points = []
        for c in s:
            if c == 'F':
                alice_points.append(0)
            elif c == 'W':
                alice_points.append(1)
            elif c == 'E':
                alice_points.append(2)
        
        # Initialize DP table
        # dp[i][j][k] represents the number of sequences for the first i rounds,
        # where j is the last move of Bob, and k is the total points difference (Bob - Alice)
        # Since the difference can be negative, we shift it by n to make it non-negative
        # The maximum possible difference is n (Bob wins all rounds), minimum is -n (Alice wins all rounds)
        # So we need a range of 2n + 1
        shift = n
        dp = [ [ [0] * (2 * n + 1) for _ in range(3) ] for _ in range(n+1) ]
        
        # Base case: no rounds, no moves, difference is 0
        dp[0][0][shift] = 1
        dp[0][1][shift] = 1
        dp[0][2][shift] = 1
        
        for i in range(1, n+1):
            for last_bob in range(3):
                for diff in range(-n, n+1):
                    if dp[i-1][last_bob][diff + shift] == 0:
                        continue
                    for current_bob in range(3):
                        if current_bob == last_bob:
                            continue
                        # Calculate the points difference for this round
                        # Bob's move is current_bob, Alice's move is alice_points[i-1]
                        # Determine who wins the round
                        bob_move = current_bob
                        alice_move = alice_points[i-1]
                        if bob_move == 0 and alice_move == 2:
                            delta = 1
                        elif bob_move == 1 and alice_move == 0:
                            delta = 1
                        elif bob_move == 2 and alice_move == 1:
                            delta = 1
                        elif bob_move == alice_move:
                            delta = 0
                        else:
                            delta = -1
                        new_diff = diff + delta
                        if -n <= new_diff <= n:
                            dp[i][current_bob][new_diff + shift] = (dp[i][current_bob][new_diff + shift] + dp[i-1][last_bob][diff + shift]) % MOD
        
        # Sum all sequences where the total difference is positive
        total = 0
        for last_bob in range(3):
            for diff in range(1, n+1):
                total = (total + dp[n][last_bob][diff + shift]) % MOD
        return total