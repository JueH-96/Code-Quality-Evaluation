class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        mod = 10**9 + 7
        
        def winner(a, b):
            if a == b:
                return 0
            if (a == 'F' and b == 'E') or \
               (a == 'W' and b == 'F') or \
               (a == 'E' and b == 'W'):
                return 1
            return -1

        def solve():
            dp = {}

            def count_wins(alice_idx, bob_prev, alice_score, bob_score):
                if alice_idx == n:
                    return 1 if bob_score > alice_score else 0

                state = (alice_idx, bob_prev, alice_score, bob_score)
                if state in dp:
                    return dp[state]

                ans = 0
                for bob_curr in ['F', 'W', 'E']:
                    if bob_curr != bob_prev:
                        alice_wins = winner(s[alice_idx], bob_curr)
                        new_alice_score = alice_score
                        new_bob_score = bob_score
                        if alice_wins == 1:
                            new_alice_score += 1
                        elif alice_wins == -1:
                            new_bob_score += 1
                        
                        ans = (ans + count_wins(alice_idx + 1, bob_curr, new_alice_score, new_bob_score)) % mod

                dp[state] = ans
                return ans

            return count_wins(0, '', 0, 0)

        return solve()