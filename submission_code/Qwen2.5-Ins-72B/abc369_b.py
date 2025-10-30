# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = [int(data[2*i+1]) for i in range(N)]
    S = [data[2*i+2] for i in range(N)]
    
    dp = [[float('inf')] * 101 for _ in range(101)]
    for i in range(101):
        for j in range(101):
            dp[i][j] = 0
    
    for i in range(N):
        next_dp = [[float('inf')] * 101 for _ in range(101)]
        for left in range(101):
            for right in range(101):
                if S[i] == 'L':
                    next_dp[A[i]][right] = min(next_dp[A[i]][right], dp[left][right] + abs(A[i] - left))
                else:
                    next_dp[left][A[i]] = min(next_dp[left][A[i]], dp[left][right] + abs(A[i] - right))
        dp = next_dp
    
    result = min(min(dp[i]) for i in range(101))
    print(result)

if __name__ == "__main__":
    main()