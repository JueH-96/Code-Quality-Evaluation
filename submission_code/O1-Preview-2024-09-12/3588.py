class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        creatures = ['F', 'W', 'E']
        # Mapping of creature beats which creature
        beats = {
            'F': 'E',  # Fire beats Earth
            'W': 'F',  # Water beats Fire
            'E': 'W',  # Earth beats Water
        }

        # Precompute the result for all combinations
        def result(a, b):
            if a == b:
                return 0  # tie
            elif beats[a] == b:
                return 1  # Alice wins
            else:
                return -1  # Bob wins

        # Initialize dp array
        max_diff = n
        dp_current = [ [0] * (2 * max_diff + 1) for _ in range(4)]  # last creature index 0-3 (including None)
        dp_next = [ [0] * (2 * max_diff + 1) for _ in range(4)]
        index_map = {None: 0, 'F': 1, 'W': 2, 'E': 3}
        dp_current[0][max_diff] = 1  # Start with last creature None and score_diff 0

        for i in range(n):
            s_i = s[i]
            # Reset dp_next
            for l in range(4):
                for d in range(2 * max_diff +1):
                    dp_next[l][d] = 0
            for last_idx in range(4):  # last_idx: 0 for None, 1 for 'F', 2 for 'W', 3 for 'E'
                for score_idx in range(2 * max_diff +1):
                    count = dp_current[last_idx][score_idx]
                    if count > 0:
                        last_creature = None if last_idx == 0 else creatures[last_idx -1]
                        for c in creatures:
                            if c != last_creature:
                                outcome = result(s_i, c)
                                actual_score_diff = score_idx - max_diff
                                new_score_diff = actual_score_diff - outcome  # Because Alice gets point when outcome==1
                                if -max_diff <= new_score_diff <= max_diff:
                                    new_score_idx = new_score_diff + max_diff
                                    next_idx = index_map[c]
                                    dp_next[next_idx][new_score_idx] = (dp_next[next_idx][new_score_idx] + count) % MOD
            # Swap dp_current and dp_next
            dp_current, dp_next = dp_next, dp_current

        # Sum over dp_current for score_diff > 0
        total = 0
        for last_idx in range(1,4):  # last creature cannot be None at the end
            for score_idx in range(max_diff +1, 2*max_diff +1):
                total = (total + dp_current[last_idx][score_idx]) % MOD

        return total