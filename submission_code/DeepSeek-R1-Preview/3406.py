class Solution:
    def numberOfStableArrays(self, zero: int, one: int, limit: int) -> int:
        MOD = 10**9 + 7

        # Initialize a 4D DP array
        # Dimensions: (zero+1) x (one+1) x 3 (for last: 0, 1, 2) x (limit+1)
        dp = [[[[0] * (limit + 1) for _ in range(3)] for __ in range(one + 1)] for ___ in range(zero + 1)]
        # Base case: 0 zeros, 0 ones, last is None (represented by 2), streak is 0
        dp[0][0][2][0] = 1

        # Process states in order of increasing sum s = z + o
        for s in range(0, zero + one + 1):
            for z in range(0, zero + 1):
                o = s - z
                if o < 0 or o > one:
                    continue

                for last in [0, 1, 2]:
                    for streak in range(0, limit + 1):
                        current = dp[z][o][last][streak]
                        if current == 0:
                            continue

                        # Try adding 0
                        if z < zero:
                            if last == 0:
                                new_streak = streak + 1
                                if new_streak > limit:
                                    continue
                                new_last = 0
                            else:
                                new_streak = 1
                                new_last = 0
                            new_z = z + 1
                            new_o = o
                            dp[new_z][new_o][new_last][new_streak] = (dp[new_z][new_o][new_last][new_streak] + current) % MOD

                        # Try adding 1
                        if o < one:
                            if last == 1:
                                new_streak = streak + 1
                                if new_streak > limit:
                                    continue
                                new_last = 1
                            else:
                                new_streak = 1
                                new_last = 1
                            new_z = z
                            new_o = o + 1
                            dp[new_z][new_o][new_last][new_streak] = (dp[new_z][new_o][new_last][new_streak] + current) % MOD

        # Sum all valid states for the final array
        total = 0
        for last in [0, 1]:
            for streak in range(1, limit + 1):
                total = (total + dp[zero][one][last][streak]) % MOD

        return total