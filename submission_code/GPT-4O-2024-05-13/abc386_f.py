# YOUR CODE HERE
import sys
input = sys.stdin.read

def min_operations(S, T):
    len_s = len(S)
    len_t = len(T)
    
    # Initialize the DP table
    dp = [[0] * (len_t + 1) for _ in range(len_s + 1)]
    
    # Base cases
    for i in range(len_s + 1):
        dp[i][0] = i
    for j in range(len_t + 1):
        dp[0][j] = j
    
    # Fill the DP table
    for i in range(1, len_s + 1):
        for j in range(1, len_t + 1):
            if S[i - 1] == T[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
    
    return dp[len_s][len_t]

def main():
    data = input().split()
    K = int(data[0])
    S = data[1]
    T = data[2]
    
    operations_needed = min_operations(S, T)
    
    if operations_needed <= K:
        print("Yes")
    else:
        print("No")

if __name__ == "__main__":
    main()