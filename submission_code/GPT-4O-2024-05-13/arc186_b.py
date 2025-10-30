# YOUR CODE HERE
import sys
input = sys.stdin.read

MOD = 998244353

def count_valid_permutations(N, A):
    dp = [0] * (N + 1)
    dp[0] = 1
    
    for i in range(1, N + 1):
        sum_dp = 0
        for j in range(i):
            if A[i - 1] <= j:
                sum_dp += dp[j]
                sum_dp %= MOD
        dp[i] = sum_dp
    
    return dp[N]

def main():
    data = input().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    result = count_valid_permutations(N, A)
    print(result)

if __name__ == "__main__":
    main()