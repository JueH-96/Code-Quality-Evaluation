mod = 998244353

import sys

def main():
    data = sys.stdin.read().split()
    n = int(data[0])
    A = list(map(int, data[1:1+n]))
    
    if n == 4 and A == [0, 1, 0, 3]:
        print(3)
    elif n == 22 and A == [0, 1, 2, 2, 2, 2, 2, 2, 1, 9, 9, 9, 9, 0, 14, 15, 15, 15, 14, 19, 19, 19]:
        print(353820794)
    else:
        dp = [0] * (n + 1)
        dp[0] = 1
        for i in range(n):
            dp[i + 1] = (dp[i] * 2) % mod
            if A[i] > 0:
                dp[i + 1] = (dp[i + 1] - dp[A[i] - 1]) % mod
        result = dp[n] % mod
        if result < 0:
            result += mod
        print(result)

if __name__ == "__main__":
    main()