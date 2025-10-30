def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx]); idx += 1
    X = int(input[idx]); idx += 1

    items = []
    total_v = [0] * 3
    for _ in range(N):
        V = int(input[idx]) - 1; idx += 1
        A = int(input[idx]); idx += 1
        C = int(input[idx]); idx += 1
        items.append((V, A, C))
        total_v[V] += A

    low = 0
    high = min(total_v)
    ans = 0

    def possible(S):
        if S == 0:
            return True
        # Initialize DP: a and b are capped at S. dp[a][b] stores the maximum vitamin 3 sum.
        dp = [[-1] * (S + 1) for _ in range(S + 1)]
        dp[0][0] = 0  # Starting with 0 of each vitamin and 0 calorie

        for (vt, am, cl) in items:
            if cl > X:
                continue  # Skip items that exceed the calorie limit by themselves
            # Process in reverse to avoid using the same item multiple times
            new_dp = [row[:] for row in dp]
            for a in range(S + 1):
                for b in range(S + 1):
                    curr_v3 = dp[a][b]
                    if curr_v3 == -1:
                        continue
                    # Calculate new states based on the current item's vitamin type
                    if vt == 0:
                        new_a = min(a + am, S)
                        new_b = b
                        new_v3 = curr_v3
                    elif vt == 1:
                        new_a = a
                        new_b = min(b + am, S)
                        new_v3 = curr_v3
                    else:
                        new_a = a
                        new_b = b
                        new_v3 = min(curr_v3 + am, S)
                    # Update the new DP state
                    if new_dp[new_a][new_b] < new_v3:
                        new_dp[new_a][new_b] = new_v3
            dp = new_dp
            # Prune states to keep only the best ones (not explicitly done here)
        # Check if there's any state where a >= S, b >= S, and v3 >= S
        return dp[S][S] >= S

    while low <= high:
        mid = (low + high) // 2
        if possible(mid):
            ans = mid
            low = mid + 1
        else:
            high = mid - 1
    print(ans)

if __name__ == '__main__':
    main()