class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        if n == 0:
            return 0
        
        # Initialize DP: each entry is a dictionary for each character, mapping to the count of D values
        dp = [{} for _ in range(n)]
        
        # Initialize the first round (i=0)
        alice_char = s[0]
        for bob_char in ['F', 'W', 'E']:
            delta = self.compute_delta(bob_char, alice_char)
            if bob_char not in dp[0]:
                dp[0][bob_char] = {}
            dp[0][bob_char][delta] = 1
        
        # Process the remaining rounds
        for i in range(1, n):
            current_alice_char = s[i]
            # Iterate through each possible previous character and their D values
            for prev_char in list(dp[i-1].keys()):
                for d_prev in list(dp[i-1][prev_char].keys()):
                    count_prev = dp[i-1][prev_char][d_prev]
                    # Try all possible next characters that are not the same as prev_char
                    for next_char in ['F', 'W', 'E']:
                        if next_char == prev_char:
                            continue
                        delta = self.compute_delta(next_char, current_alice_char)
                        new_d = d_prev + delta
                        # Update the current state
                        if next_char not in dp[i]:
                            dp[i][next_char] = {}
                        if new_d not in dp[i][next_char]:
                            dp[i][next_char][new_d] = 0
                        dp[i][next_char][new_d] = (dp[i][next_char][new_d] + count_prev) % MOD
        
        # Sum all valid counts where D > 0
        total = 0
        for char in ['F', 'W', 'E']:
            if char in dp[-1]:
                for d in dp[-1][char]:
                    if d > 0:
                        total = (total + dp[-1][char][d]) % MOD
        return total
    
    def compute_delta(self, bob_char, alice_char):
        if bob_char == alice_char:
            return 0
        # Check if Bob's choice beats Alice's
        if (bob_char == 'F' and alice_char == 'E') or \
           (bob_char == 'W' and alice_char == 'F') or \
           (bob_char == 'E' and alice_char == 'W'):
            return 1
        else:
            return -1