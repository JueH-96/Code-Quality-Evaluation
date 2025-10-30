def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    # We need to maximize the sum of the sequence S
    # We can use a dynamic programming approach to solve this problem
    # dp[i] will represent the maximum sum we can achieve using the first i elements of A
    
    # Initialize dp array
    dp = [0] * (N + 1)
    
    # Base case: if no elements are taken, the sum is 0
    dp[0] = 0
    
    # Fill the dp array
    for i in range(1, N + 1):
        # We have two choices:
        # 1. Take the current element A[i-1] and add it to the sum we had until dp[i-1]
        # 2. Do not take the current element, which means we just carry forward the previous sum
        # Since we want to maximize the sum, we take the maximum of these two choices
        dp[i] = max(dp[i-1] + A[i-1], dp[i-1])
    
    # The answer is the maximum sum we can achieve using all N elements
    print(dp[N])

if __name__ == "__main__":
    main()