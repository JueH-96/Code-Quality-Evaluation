class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # dp[i][j] will store the number of ways Bob can have a sequence ending with creature j
        # after i rounds such that Bob is winning.
        # j = 0 for Fire Dragon, 1 for Water Serpent, 2 for Earth Golem
        dp = [[0] * 3 for _ in range(n + 1)]
        
        # Initial conditions: Bob has no moves yet, so no points
        # dp[0][j] = 0 for all j, but we need to initialize the first move
        # Bob can start with any creature
        dp[0][0] = dp[0][1] = dp[0][2] = 1
        
        # Points mapping
        # Alice vs Bob: (Alice's creature, Bob's creature) -> Bob's point
        points = {
            ('F', 'W'): 1,
            ('W', 'E'): 1,
            ('E', 'F'): 1,
            ('F', 'E'): -1,
            ('W', 'F'): -1,
            ('E', 'W'): -1,
            ('F', 'F'): 0,
            ('W', 'W'): 0,
            ('E', 'E'): 0
        }
        
        for i in range(1, n + 1):
            alice_creature = s[i - 1]
            for bob_creature in range(3):
                # Calculate the points for this round
                if bob_creature == 0:
                    bob_creature_char = 'F'
                elif bob_creature == 1:
                    bob_creature_char = 'W'
                else:
                    bob_creature_char = 'E'
                
                # Calculate points for this round
                point = points[(alice_creature, bob_creature_char)]
                
                # Transition from previous states
                for prev_bob_creature in range(3):
                    if prev_bob_creature != bob_creature:
                        # If Bob's previous creature is different, we can transition
                        dp[i][bob_creature] += dp[i - 1][prev_bob_creature]
                        dp[i][bob_creature] %= MOD
                
                # Adjust the score based on the current round's result
                if point == 1:
                    dp[i][bob_creature] += 1
                elif point == -1:
                    dp[i][bob_creature] -= 1
                
                dp[i][bob_creature] %= MOD
        
        # Count all sequences where Bob's score is greater than Alice's
        total_ways = 0
        for j in range(3):
            total_ways += dp[n][j]
            total_ways %= MOD
        
        return total_ways