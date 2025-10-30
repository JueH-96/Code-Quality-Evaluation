import sys
from collections import defaultdict

def main():
    n = int(sys.stdin.readline())
    a = list(map(int, sys.stdin.readline().split()))
    last = defaultdict(int)
    dp = [0]*(n+1)
    for i in range(n):
        dp[i+1] = dp[i] + i - last[a[i]] + 1
        last[a[i]] = i + 1
    print(dp[-1])

if __name__ == "__main__":
    main()