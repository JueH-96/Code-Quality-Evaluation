import sys
from collections import deque

def min_cost(T, N, bags):
    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0
    for i in range(len(T)):
        for j in range(N):
            for k in range(1, bags[j][0] + 1):
                if i + k <= len(T) and T[i:i + k] in bags[j][1]:
                    dp[i + k] = min(dp[i + k], dp[i] + 1)
    return dp[-1] if dp[-1] != float('inf') else -1

def main():
    T = sys.stdin.readline().strip()
    N = int(sys.stdin.readline().strip())
    bags = []
    for _ in range(N):
        line = sys.stdin.readline().strip().split()
        bags.append((int(line[0]), set(line[1:])))
    print(min_cost(T, N, bags))

if __name__ == "__main__":
    main()