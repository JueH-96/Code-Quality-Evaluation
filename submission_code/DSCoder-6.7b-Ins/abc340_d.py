import sys

def read_input():
    N = int(sys.stdin.readline().strip())
    stages = [list(map(int, sys.stdin.readline().strip().split())) for _ in range(N-1)]
    return N, stages

def solve(N, stages):
    dp = [0] * N
    dp[0] = stages[0][0]
    for i in range(1, N-1):
        dp[i] = min(dp[i-1] + stages[i-1][0], dp[stages[i-1][2]-2] + stages[i-1][1])
    dp[-1] = min(dp[-2] + stages[-1][0], dp[stages[-1][2]-2] + stages[-1][1])
    return dp[-1]

def main():
    N, stages = read_input()
    print(solve(N, stages))

if __name__ == "__main__":
    main()