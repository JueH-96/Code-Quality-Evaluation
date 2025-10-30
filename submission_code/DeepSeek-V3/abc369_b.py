# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = []
    S = []
    for i in range(N):
        A.append(int(data[2*i+1]))
        S.append(data[2*i+2])
    
    # Initialize DP table
    # dp[i][l][r] represents the minimum fatigue after the i-th press, with left hand on l and right hand on r
    # Initialize with infinity
    INF = float('inf')
    dp = [[[INF for _ in range(101)] for _ in range(101)] for _ in range(N+1)]
    
    # Initial state: before any press, hands can be on any keys
    for l in range(1, 101):
        for r in range(1, 101):
            dp[0][l][r] = 0
    
    for i in range(1, N+1):
        a = A[i-1]
        s = S[i-1]
        for l_prev in range(1, 101):
            for r_prev in range(1, 101):
                if dp[i-1][l_prev][r_prev] == INF:
                    continue
                if s == 'L':
                    # Move left hand to a
                    new_l = a
                    new_r = r_prev
                    fatigue = dp[i-1][l_prev][r_prev] + abs(new_l - l_prev)
                    if fatigue < dp[i][new_l][new_r]:
                        dp[i][new_l][new_r] = fatigue
                else:
                    # Move right hand to a
                    new_l = l_prev
                    new_r = a
                    fatigue = dp[i-1][l_prev][r_prev] + abs(new_r - r_prev)
                    if fatigue < dp[i][new_l][new_r]:
                        dp[i][new_l][new_r] = fatigue
    
    # Find the minimum fatigue after all presses
    min_fatigue = INF
    for l in range(1, 101):
        for r in range(1, 101):
            if dp[N][l][r] < min_fatigue:
                min_fatigue = dp[N][l][r]
    
    print(min_fatigue)

if __name__ == "__main__":
    main()