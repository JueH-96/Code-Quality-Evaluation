def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    # Use two states:
    # state 0: healthy, state 1: upset
    # Initialize dp with dp[0] = 0 (healthy) and dp[1] = -infinity (cannot be upset initially).
    INF_NEG = -10**18  # sufficiently small negative value given constraints
    dp = [0, INF_NEG]

    # Process each course one by one.
    for _ in range(N):
        line = input().strip()
        if not line:
            continue
        X_str, Y_str = line.split()
        X = int(X_str)
        Y = int(Y_str)
        
        # Prepare new state values for the next course.
        new_dp = [INF_NEG, INF_NEG]
        
        # For each of the current states, try both actions: skip and eat.
        for state in (0, 1):
            current_score = dp[state]
            # If the current state is unreachable, skip.
            if current_score == INF_NEG:
                continue
            # 1. Option: Skip the current course.
            #    The state remains unchanged.
            if state == 0:
                new_dp[0] = max(new_dp[0], current_score)
            else:
                new_dp[1] = max(new_dp[1], current_score)
            
            # 2. Option: Eat the current course.
            if state == 0:  # currently healthy
                if X == 0:
                    # antidotal course: healthy remains healthy.
                    new_dp[0] = max(new_dp[0], current_score + Y)
                else:
                    # poisonous course: healthy becomes upset.
                    new_dp[1] = max(new_dp[1], current_score + Y)
            else:  # currently upset
                if X == 0:
                    # antidotal course: upset becomes healthy.
                    new_dp[0] = max(new_dp[0], current_score + Y)
                else:
                    # poisonous course: he dies. So, do nothing.
                    pass
        
        dp = new_dp
    
    # At the end, he must be alive. Both healthy and upset states are safe.
    res = max(dp[0], dp[1])
    # If it's optimal to eat nothing (i.e. result is negative), answer is 0.
    if res < 0:
        res = 0
    sys.stdout.write(str(res))


if __name__ == '__main__':
    main()