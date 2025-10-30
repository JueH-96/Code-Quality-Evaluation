class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        
        # dp[i][j] represents the number of valid sequences for Bob up to the i-th round
        # where the i-th creature is j
        # j = 0 -> 'F', j = 1 -> 'W', j = 2 -> 'E'
        dp = [[0] * 3 for _ in range(n + 1)]
        
        # Initialize the first round based on what Alice summons
        for j in range(3):
            if self.beats(j, s[0]):
                dp[1][j] = 1
        
        # Fill dp table
        for i in range(2, n + 1):
            for j in range(3):  # Bob's i-th move
                if self.beats(j, s[i-1]):  # Only consider if Bob can beat Alice in this round
                    for k in range(3):  # Bob's previous move
                        if k != j:  # Bob cannot summon the same creature consecutively
                            dp[i][j] = (dp[i][j] + dp[i-1][k]) % MOD
        
        # Sum all valid sequences that end with any creature
        result = sum(dp[n]) % MOD
        return result
    
    def beats(self, bob, alice):
        # Convert index to creature
        # 0 -> 'F', 1 -> 'W', 2 -> 'E'
        bob_creature = ['F', 'W', 'E'][bob]
        
        # Determine if Bob's creature beats Alice's creature
        if bob_creature == 'F' and alice == 'E':
            return True
        elif bob_creature == 'W' and alice == 'F':
            return True
        elif bob_creature == 'E' and alice == 'W':
            return True
        return False