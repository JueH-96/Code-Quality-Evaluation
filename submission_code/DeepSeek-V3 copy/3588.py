MOD = 10**9 + 7

class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        # Precompute the points Alice gets for each round
        alice_points = 0
        for c in s:
            if c == 'F':
                alice_points += 0  # No points for Alice in this case
            elif c == 'W':
                alice_points += 0
            elif c == 'E':
                alice_points += 0
        # We need to find the number of sequences Bob can have such that his total points > alice_points
        # and no two consecutive moves are the same.
        # We will use dynamic programming to count the number of valid sequences for Bob
        # and also track the points difference.
        # dp[i][last_move][diff] = number of ways to reach round i with last_move and points difference diff
        # Initialize dp[0][move][0] = 1 for all moves
        # Then for each round, update the dp based on the previous state
        # Finally, sum all dp[n][move][diff] where diff > alice_points
        # Initialize the DP table
        # Since the points difference can be negative, we need to shift the difference to be non-negative
        # The maximum possible difference is n (if Bob wins all rounds), and minimum is -n (if Alice wins all rounds)
        # So we can shift the difference by n to make it non-negative
        shift = n
        max_diff = 2 * n
        dp = [ [ [0] * (max_diff + 1) for _ in range(3) ] for _ in range(n+1) ]
        # Initialize the first round
        for move in range(3):
            dp[0][move][0 + shift] = 1
        # Iterate through each round
        for i in range(n):
            for last_move in range(3):
                for diff in range(-n, n+1):
                    if dp[i][last_move][diff + shift] == 0:
                        continue
                    # Current move for Bob can be any of the three, but not the same as last_move
                    for current_move in range(3):
                        if current_move == last_move:
                            continue
                        # Calculate the points difference for this move
                        # Determine the outcome of the round
                        alice_move = s[i]
                        if alice_move == 'F':
                            if current_move == 2:  # Bob's Earth Golem
                                # Alice's Fire Dragon vs Bob's Earth Golem: Bob gets a point
                                new_diff = diff + 1
                            elif current_move == 1:  # Bob's Water Serpent
                                # Alice's Fire Dragon vs Bob's Water Serpent: Alice gets a point
                                new_diff = diff - 1
                            else:
                                # Alice's Fire Dragon vs Bob's Fire Dragon: no points
                                new_diff = diff
                        elif alice_move == 'W':
                            if current_move == 0:  # Bob's Fire Dragon
                                # Alice's Water Serpent vs Bob's Fire Dragon: Bob gets a point
                                new_diff = diff + 1
                            elif current_move == 2:  # Bob's Earth Golem
                                # Alice's Water Serpent vs Bob's Earth Golem: Alice gets a point
                                new_diff = diff - 1
                            else:
                                # Alice's Water Serpent vs Bob's Water Serpent: no points
                                new_diff = diff
                        elif alice_move == 'E':
                            if current_move == 1:  # Bob's Water Serpent
                                # Alice's Earth Golem vs Bob's Water Serpent: Bob gets a point
                                new_diff = diff + 1
                            elif current_move == 0:  # Bob's Fire Dragon
                                # Alice's Earth Golem vs Bob's Fire Dragon: Alice gets a point
                                new_diff = diff - 1
                            else:
                                # Alice's Earth Golem vs Bob's Earth Golem: no points
                                new_diff = diff
                        # Update the DP table
                        if new_diff + shift < 0 or new_diff + shift > max_diff:
                            continue
                        dp[i+1][current_move][new_diff + shift] = (dp[i+1][current_move][new_diff + shift] + dp[i][last_move][diff + shift]) % MOD
        # Now, sum all dp[n][move][diff] where diff > alice_points
        total = 0
        for move in range(3):
            for diff in range(alice_points + 1, n+1):
                total = (total + dp[n][move][diff + shift]) % MOD
        return total