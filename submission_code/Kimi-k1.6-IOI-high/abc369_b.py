def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    notes = []
    for _ in range(N):
        a = int(data[idx])
        s = data[idx + 1]
        notes.append((a, s))
        idx += 2
    
    INF = float('inf')
    prev_dp = [[INF] * 101 for _ in range(101)]
    
    # Initialize DP table for step 0 (before any keys are pressed)
    for l in range(1, 101):
        for r in range(1, 101):
            prev_dp[l][r] = 0
    
    for a, s in notes:
        curr_dp = [[INF] * 101 for _ in range(101)]
        for l_prev in range(1, 101):
            for r_prev in range(1, 101):
                if prev_dp[l_prev][r_prev] == INF:
                    continue
                if s == 'L':
                    new_l = a
                    new_r = r_prev
                    cost = prev_dp[l_prev][r_prev] + abs(new_l - l_prev)
                    if cost < curr_dp[new_l][new_r]:
                        curr_dp[new_l][new_r] = cost
                else:
                    new_r = a
                    new_l = l_prev
                    cost = prev_dp[l_prev][r_prev] + abs(new_r - r_prev)
                    if cost < curr_dp[new_l][new_r]:
                        curr_dp[new_l][new_r] = cost
        prev_dp = curr_dp
    
    min_fatigue = INF
    for l in range(1, 101):
        for r in range(1, 101):
            if prev_dp[l][r] < min_fatigue:
                min_fatigue = prev_dp[l][r]
    print(min_fatigue)

if __name__ == "__main__":
    main()