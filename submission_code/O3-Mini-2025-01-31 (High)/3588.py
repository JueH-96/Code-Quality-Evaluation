from collections import defaultdict

MOD = 10**9+7

class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        # Map characters to indices:
        # 0: 'F', 1: 'W', 2: 'E'
        move_to_idx = {'F': 0, 'W': 1, 'E': 2}
        # outcome_table[a][b] gives the result (for Bob) when:
        #   a is Alice's move (index) and b is Bob's move (index).
        # The rules are cyclic:
        # • If one summons Fire Dragon (F) and the other Earth Golem (E): the one with F wins.
        # • If one summons Water Serpent (W) and the other Fire Dragon (F): the one with W wins.
        # • If one summons Earth Golem (E) and the other Water Serpent (W): the one with E wins.
        # Thus for Bob:
        # If Alice uses F (0): Bob wins if he uses W (1), loses if he uses E (2), tie if Bob also uses F.
        # If Alice uses W (1): Bob wins if he uses E (2), loses if he uses F (0), tie if Bob also uses W.
        # If Alice uses E (2): Bob wins if he uses F (0), loses if he uses W (1), tie if Bob also uses E.
        outcome_table = [
            [0, 1, -1],   # Alice: F (0)
            [-1, 0, 1],   # Alice: W (1)
            [1, -1, 0]    # Alice: E (2)
        ]
        
        # We'll use dynamic programming.
        # Let dp[b] be a dictionary for Bob’s last move b (0,1,2).
        # Each dictionary maps a score difference (Bob's wins minus Alice's wins) to a count:
        #   dp[b][d] = number of ways to get difference d ending with move b.
        dp = [defaultdict(int) for _ in range(3)]
        
        # For round 0, Bob may choose any creature.
        a0 = move_to_idx[s[0]]
        for bob in range(3):
            result = outcome_table[a0][bob]
            dp[bob][result] = (dp[bob][result] + 1) % MOD
        
        # Process rounds 1 to n-1.
        for i in range(1, n):
            a_move = move_to_idx[s[i]]
            new_dp = [defaultdict(int) for _ in range(3)]
            # Bob is not allowed to repeat the same move as the previous round.
            for prev in range(3):
                for diff, count in dp[prev].items():
                    # Only consider if ways are nonzero.
                    if count:
                        for bob in range(3):
                            if bob == prev:
                                continue
                            outcome = outcome_table[a_move][bob]
                            new_diff = diff + outcome
                            new_dp[bob][new_diff] = (new_dp[bob][new_diff] + count) % MOD
            dp = new_dp
        
        # Sum over all sequences where Bob's overall score is > Alice's (i.e. diff > 0).
        ans = 0
        for bob in range(3):
            for diff, count in dp[bob].items():
                if diff > 0:
                    ans = (ans + count) % MOD
        return ans

# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    s1 = "FFF"
    print(sol.countWinningSequences(s1))  # Expected output: 3
    # Example 2:
    s2 = "FWEFW"
    print(sol.countWinningSequences(s2))  # Expected output: 18