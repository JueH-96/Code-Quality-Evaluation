def main():
    import sys
    input = sys.stdin.read().split()
    idx = 0
    N = int(input[idx])
    idx += 1
    presses = []
    for _ in range(N):
        a = int(input[idx])
        s = input[idx + 1]
        presses.append((a, s))
        idx += 2
    
    # Initialize DP
    prev_dp = [[0] * 101 for _ in range(101)]
    
    for a, s in presses:
        curr_dp = [[float('inf')] * 101 for _ in range(101)]
        for l_prev in range(101):
            for r_prev in range(101):
                if prev_dp[l_prev][r_prev] == float('inf'):
                    continue
                if s == 'L':
                    new_l = a
                    new_r = r_prev
                    cost = prev_dp[l_prev][r_prev] + abs(a - l_prev)
                    if cost < curr_dp[new_l][new_r]:
                        curr_dp[new_l][new_r] = cost
                else:
                    new_r = a
                    new_l = l_prev
                    cost = prev_dp[l_prev][r_prev] + abs(a - r_prev)
                    if cost < curr_dp[new_l][new_r]:
                        curr_dp[new_l][new_r] = cost
        prev_dp = curr_dp
    
    min_cost = min(min(row) for row in prev_dp)
    print(min_cost)

if __name__ == "__main__":
    main()