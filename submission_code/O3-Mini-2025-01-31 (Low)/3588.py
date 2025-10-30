MOD = 10**9 + 7

class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        # Map creatures to indices: 0: 'F', 1: 'W', 2: 'E'
        # Outcome if Bob uses move b against Alice's move a:
        # Bob wins +1 if:
        #   (b = 'F' and a = 'E'), (b = 'W' and a = 'F'), (b = 'E' and a = 'W')
        # Bob loses (-1 point) if:
        #   (b = 'E' and a = 'F'), (b = 'F' and a = 'W'), (b = 'W' and a = 'E')
        # Otherwise outcome is 0.
        # We'll create a 3x3 table outcome[b][a]
        outcome = [[0] * 3 for _ in range(3)]
        # For clarity, assign indices: F:0, W:1, E:2.
        # Set win conditions:
        outcome[0][2] = 1   # Bob F vs Alice E -> Bob wins
        outcome[1][0] = 1   # Bob W vs Alice F
        outcome[2][1] = 1   # Bob E vs Alice W
        # Set lose conditions:
        outcome[2][0] = -1  # Bob E vs Alice F -> Alice wins
        outcome[0][1] = -1  # Bob F vs Alice W
        outcome[1][2] = -1  # Bob W vs Alice E
        
        # Map from char to index for Alice
        ca = {'F': 0, 'W': 1, 'E': 2}
        
        # dp[i][b] where b is bob's last move index, and dp[i][b] is a list that tracks count for each possible difference
        # We will offset the difference by n
        offset = n
        size = 2 * n + 1  # difference range: -n ... n
        
        # Initialize dp as a list of 3 lists (for Bob's last move) for round i.
        dp = [[0] * size for _ in range(3)]
        # For round 0 (first round), Bob can choose any move.
        alice0 = ca[s[0]]
        for b in range(3):
            diff = outcome[b][alice0]  # difference after this round
            dp[b][diff + offset] = (dp[b][diff + offset] + 1) % MOD
        
        # Process rounds from 1 to n-1
        for i in range(1, n):
            # new dp for round i, reinitialize dp_next
            dp_next = [[0] * size for _ in range(3)]
            alice_move = ca[s[i]]
            for prev_move in range(3):
                for diff in range(size):
                    ways = dp[prev_move][diff]
                    if ways:
                        # Next Bob's move must be different from prev_move
                        for cur_move in range(3):
                            if cur_move == prev_move:
                                continue
                            # outcome of this round
                            delta = outcome[cur_move][alice_move]
                            new_diff = diff - offset + delta  # actual diff (not offset)
                            if -n <= new_diff <= n:
                                dp_next[cur_move][new_diff + offset] = (dp_next[cur_move][new_diff + offset] + ways) % MOD
            dp = dp_next
        
        # Sum over all sequences that result in Bob having strictly more points than Alice (i.e., difference > 0)
        res = 0
        for move in range(3):
            for d in range(offset+1, size):  # d-offset > 0 => d > offset
                res = (res + dp[move][d]) % MOD
        return res
                        
# For testing
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    s1 = "FFF"
    print(sol.countWinningSequences(s1))  # Expected output: 3
    # Example 2:
    s2 = "FWEFW"
    print(sol.countWinningSequences(s2))  # Expected output: 18