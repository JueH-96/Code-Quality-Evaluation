# YOUR CODE HERE

import sys

MOD = 998244353

def solve(n, m):
    if n == 1:
        return m
    else:
        return (m * (m - 1) ** (n - 1)) % MOD

def main():
    n, m = map(int, sys.stdin.readline().split())
    print(solve(n, m))

if __name__ == "__main__":
    main()