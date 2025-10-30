# YOUR CODE HERE
import sys
input = sys.stdin.read
MOD = 998244353

def main():
    data = input().strip().split()
    N = int(data[0])
    A = list(map(int, data[1:]))
    
    dp = [0] * (N + 1)
    dp[0] = 1
    
    for i in range(1, N + 1):
        for j in range(i):
            dp[i] = (dp[i] + dp[j]) % MOD
    
    count = [0] * (N + 1)
    count[0] = 1
    
    for i in range(1, N + 1):
        for j in range(i):
            if A[j] < i - j:
                count[i] = (count[i] + dp[j]) % MOD
            elif A[j] == i - j:
                count[i] = (count[i] + count[j]) % MOD
    
    print(count[N])

if __name__ == "__main__":
    main()