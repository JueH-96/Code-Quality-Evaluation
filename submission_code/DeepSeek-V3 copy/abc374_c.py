# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    K = list(map(int, data[1:N+1]))
    
    total = sum(K)
    half = total // 2
    
    # Initialize DP array
    dp = [False] * (half + 1)
    dp[0] = True
    
    for k in K:
        for j in range(half, k - 1, -1):
            if dp[j - k]:
                dp[j] = True
    
    # Find the largest possible sum <= half
    max_sum = 0
    for j in range(half, -1, -1):
        if dp[j]:
            max_sum = j
            break
    
    # The minimum possible maximum is the larger of max_sum and total - max_sum
    result = max(max_sum, total - max_sum)
    print(result)

if __name__ == "__main__":
    main()