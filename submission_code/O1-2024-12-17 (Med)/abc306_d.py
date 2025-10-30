def main():
    import sys

    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    X = [int(input_data[2*i+1]) for i in range(N)]
    Y = [int(input_data[2*i+2]) for i in range(N)]

    # We'll maintain two DP states at each step:
    # dphealthy: the maximum sum of tastiness achievable if Takahashi ends this step healthy
    # dpupset:   the maximum sum of tastiness achievable if Takahashi ends this step upset
    # If it is impossible to be in a certain state, we keep it at -∞.

    NEG_INF = float('-inf')
    dphealthy = 0    # Initially healthy with sum = 0 (i.e., hasn't eaten anything yet)
    dpupset = NEG_INF

    for i in range(N):
        x = X[i]  # 0 (antidote) or 1 (poison)
        y = Y[i]

        new_healthy = NEG_INF
        new_upset = NEG_INF

        # Option 1: Skip the course (state does not change, sum is the same).
        new_healthy = max(new_healthy, dphealthy)
        new_upset = max(new_upset, dpupset)

        # Option 2: Eat the course, transition states accordingly.
        if x == 0:
            # If the course is antidotal:
            #   healthy + antidote -> remain healthy
            #   upset + antidote   -> become healthy
            if dphealthy != NEG_INF:
                new_healthy = max(new_healthy, dphealthy + y)
            if dpupset != NEG_INF:
                new_healthy = max(new_healthy, dpupset + y)
        else:
            # If the course is poisonous:
            #   healthy + poison -> become upset
            #   upset + poison  -> die (no valid transition)
            if dphealthy != NEG_INF:
                new_upset = max(new_upset, dphealthy + y)

        dphealthy, dpupset = new_healthy, new_upset

    # We must survive: final state can be either healthy or upset, but not dead.
    # Also, we can skip eating all courses, so the result must be at least 0.
    ans = max(0, dphealthy, dpupset)
    print(ans)

# Do not forget to call main at the end
if __name__ == "__main__":
    main()