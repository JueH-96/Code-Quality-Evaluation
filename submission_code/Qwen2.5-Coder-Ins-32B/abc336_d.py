# YOUR CODE HERE
import sys

def max_pyramid_sequence(N, A):
    # Initialize the dp array to store the maximum pyramid size ending at each index
    dp = [0] * N
    
    # Forward pass to calculate the longest increasing subsequence ending at each index
    for i in range(N):
        dp[i] = 1
        for j in range(i):
            if A[j] < A[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    # Backward pass to calculate the longest decreasing subsequence starting at each index
    for i in range(N-1, -1, -1):
        for j in range(i+1, N):
            if A[j] < A[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    
    # The result is the maximum value in dp array, which represents the longest pyramid sequence
    return max(dp)

def main():
    import sys
    input = sys.stdin.read().split()
    N = int(input[0])
    A = list(map(int, input[1:]))
    result = max_pyramid_sequence(N, A)
    print(result)

if __name__ == "__main__":
    main()