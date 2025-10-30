import sys

def solve():
    S = sys.stdin.readline().strip()
    MOD = 998244353

    # dp[0]: Number of ways to form a prefix such that no XX subsequence has been formed yet.
    #        (All uppercase characters encountered so far are distinct.)
    # dp[1]: Number of ways to form a prefix such that at least one XX subsequence has been formed,
    #        but no XXy subsequence has been formed yet.
    # dp[2]: Number of ways to form a prefix such that at least one XXy subsequence has been formed,
    #        but no XXyU subsequence has been formed yet.
    
    dp = [1, 0, 0] # Initial state: empty string, 1 way to be in dp[0]

    for char in S:
        next_dp = [0, 0, 0]

        uc_choices = 0 # Number of ways the current char can be an uppercase letter
        lc_choices = 0 # Number of ways the current char can be a lowercase letter

        if char == '?':
            uc_choices = 26
            lc_choices = 26
        elif char.isupper():
            uc_choices = 1
            lc_choices = 0
        elif char.islower():
            uc_choices = 0
            lc_choices = 1

        # Transitions from dp[0] (no XX subsequence yet)
        # 1. Current char becomes a lowercase letter:
        #    Does not form XX. Stays in dp[0].
        next_dp[0] = (next_dp[0] + dp[0] * lc_choices) % MOD

        # 2. Current char becomes an uppercase letter:
        #    If it forms XX (by matching a previously seen uppercase char): transitions to dp[1].
        #    This is implicitly 1 way (the specific char that forms XX).
        #    If it does not form XX (by being a new distinct uppercase char): stays in dp[0].
        #    This is (uc_choices - 1) ways. If uc_choices is 1 (specific char), this means 0 ways.
        if uc_choices > 0:
            next_dp[0] = (next_dp[0] + dp[0] * (uc_choices - 1)) % MOD # Stays in dp[0]
            next_dp[1] = (next_dp[1] + dp[0] * 1) % MOD # Transitions to dp[1]

        # Transitions from dp[1] (XX found, but no XXy yet)
        # 1. Current char becomes a lowercase letter:
        #    Completes XXy. Transitions to dp[2].
        next_dp[2] = (next_dp[2] + dp[1] * lc_choices) % MOD
        
        # 2. Current char becomes an uppercase letter:
        #    Does not form XXy. Stays in dp[1].
        next_dp[1] = (next_dp[1] + dp[1] * uc_choices) % MOD

        # Transitions from dp[2] (XXy found, but no XXyU yet)
        # 1. Current char becomes a lowercase letter:
        #    Does not complete XXyU. Stays in dp[2].
        next_dp[2] = (next_dp[2] + dp[2] * lc_choices) % MOD

        # 2. Current char becomes an uppercase letter:
        #    Completes XXyU. These strings are considered invalid (DDoS-type) and are not counted.
        #    So, no contribution from dp[2] to any next_dp state if an uppercase letter is chosen.

        dp = next_dp

    # The total number of non-DDoS-type strings is the sum of ways in all valid states.
    result = (dp[0] + dp[1] + dp[2]) % MOD
    sys.stdout.write(str(result) + '
')

solve()