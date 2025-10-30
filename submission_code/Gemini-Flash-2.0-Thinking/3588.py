class Solution:
    def countWinningSequences(self, s: str) -> int:
        n = len(s)
        mod = 10**9 + 7
        dp = {}  # (i, last_bob_move, diff) -> count

        def solve(i, last_bob_move, diff):
            if (i, last_bob_move, diff) in dp:
                return dp[(i, last_bob_move, diff)]

            if i == n:
                return 1 if diff > 0 else 0

            count = 0
            alice_move = s[i]

            for current_bob_move in ['F', 'W', 'E']:
                if current_bob_move != last_bob_move:
                    delta_score = 0
                    if current_bob_move == 'F' and alice_move == 'E':
                        delta_score += 1
                    elif current_bob_move == 'W' and alice_move == 'F':
                        delta_score += 1
                    elif current_bob_move == 'E' and alice_move == 'W':
                        delta_score += 1

                    if alice_move == 'F' and current_bob_move == 'E':
                        delta_score -= 1
                    elif alice_move == 'W' and current_bob_move == 'F':
                        delta_score -= 1
                    elif alice_move == 'E' and current_bob_move == 'W':
                        delta_score -= 1

                    count = (count + solve(i + 1, current_bob_move, diff + delta_score)) % mod

            dp[(i, last_bob_move, diff)] = count
            return count

        return solve(0, None, 0)