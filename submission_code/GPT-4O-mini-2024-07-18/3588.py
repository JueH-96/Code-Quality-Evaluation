class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # dp[i][j] will store the number of valid sequences Bob can use to beat Alice
        # up to the i-th round, where j indicates the last creature Bob summoned.
        # j = 0 -> Fire Dragon (F)
        # j = 1 -> Water Serpent (W)
        # j = 2 -> Earth Golem (E)
        dp = [[0] * 3 for _ in range(n + 1)]
        
        # Base case: before any rounds, there's one way to have no moves
        dp[0] = [1, 1, 1]  # Bob can start with any of the three creatures
        
        for i in range(1, n + 1):
            alice_move = s[i - 1]
            for j in range(3):
                # Bob's last move
                if j == 0:  # Bob summoned Fire Dragon
                    # Bob can only summon Fire Dragon if Alice summoned Earth Golem
                    if alice_move == 'E':
                        dp[i][j] = (dp[i - 1][1] + dp[i - 1][2]) % MOD  # W or E
                elif j == 1:  # Bob summoned Water Serpent
                    # Bob can only summon Water Serpent if Alice summoned Fire Dragon
                    if alice_move == 'F':
                        dp[i][j] = (dp[i - 1][0] + dp[i - 1][2]) % MOD  # F or E
                elif j == 2:  # Bob summoned Earth Golem
                    # Bob can only summon Earth Golem if Alice summoned Water Serpent
                    if alice_move == 'W':
                        dp[i][j] = (dp[i - 1][0] + dp[i - 1][1]) % MOD  # F or W
        
        # Calculate the total number of sequences Bob can use to beat Alice
        total_sequences = 0
        for j in range(3):
            total_sequences = (total_sequences + dp[n][j]) % MOD
        
        # Now we need to count the sequences that actually beat Alice
        # We can calculate the total points Alice gets
        alice_points = 0
        for move in s:
            if move == 'F':
                alice_points += 0  # F vs F = 0
            elif move == 'W':
                alice_points += 1  # W vs F = 1
            elif move == 'E':
                alice_points += 0  # E vs W = 0
        
        # Now we need to find the number of sequences that give Bob more points than Alice
        winning_sequences = 0
        
        # We need to iterate through all possible sequences of Bob's moves
        from itertools import product
        
        for moves in product('FWE', repeat=n):
            if all(moves[i] != moves[i - 1] for i in range(1, n)):  # Check no consecutive moves
                bob_points = 0
                for i in range(n):
                    if moves[i] == 'F' and s[i] == 'E':
                        bob_points += 1
                    elif moves[i] == 'W' and s[i] == 'F':
                        bob_points += 1
                    elif moves[i] == 'E' and s[i] == 'W':
                        bob_points += 1
                
                if bob_points > alice_points:
                    winning_sequences += 1
        
        return winning_sequences % MOD