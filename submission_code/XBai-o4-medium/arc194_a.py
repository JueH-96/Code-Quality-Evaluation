def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:N+1]))
    
    # Initialize dp array
    dp = [0] * (N + 2)  # dp[0] = 0, others are -infinity
    for i in range(1, N + 1):
        new_dp = [-float('inf')] * (i + 1)
        for k in range(0, i + 1):
            option1 = -float('inf')
            if k - 1 >= 0:
                option1 = dp[k-1] + A[i-1]
            option2 = -float('inf')
            if k + 1 <= i - 1 and i >= 1:
                if i - 2 >= 0:
                    option2 = dp[k+1] - A[i-2]
            new_dp[k] = max(option1, option2)
        dp = new_dp
    
    print(max(dp))

if __name__ == "__main__":
    main()