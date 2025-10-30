import sys

def read_input():
    N, W = map(int, sys.stdin.readline().split())
    items = [list(map(int, sys.stdin.readline().split())) for _ in range(N)]
    return N, W, items

def solve(N, W, items):
    dp = [0] * (W + 1)
    for w, v in items:
        for j in range(W, w - 1, -1):
            dp[j] = max(dp[j], dp[j - w] + v)
    return dp[-1]

def main():
    N, W, items = read_input()
    print(solve(N, W, items))

if __name__ == "__main__":
    main()