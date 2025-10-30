def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    n = int(input[idx])
    idx += 1
    w_total = int(input[idx])
    idx += 1

    types = []
    for _ in range(n):
        wi = int(input[idx])
        vi = int(input[idx + 1])
        types.append((wi, vi))
        idx += 2

    INF = float('-inf')
    dp = [INF] * (w_total + 1)
    dp[0] = 0

    for wi, vi in types:
        max_k = w_total // wi
        options = []
        for k in range(0, max_k + 1):
            current_w = k * wi
            current_h = vi * k - k * k
            options.append((current_w, current_h))
        
        # Create a temporary DP to base our updates on
        tmp_dp = dp.copy()
        
        # Process each option
        for w_opt, h_opt in options:
            for current_w in range(w_total, w_opt - 1, -1):
                if tmp_dp[current_w - w_opt] != INF:
                    if dp[current_w] < tmp_dp[current_w - w_opt] + h_opt:
                        dp[current_w] = tmp_dp[current_w - w_opt] + h_opt

    # Find the maximum happiness in dp
    max_happiness = max([x for x in dp if x != INF])
    print(max_happiness)

if __name__ == '__main__':
    main()