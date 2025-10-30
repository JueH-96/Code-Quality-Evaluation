import sys

def solve():
    T = input().strip()
    N = int(input().strip())
    bags = []
    for _ in range(N):
        line = input().strip().split()
        A_i = int(line[0])
        strings = line[1:]
        bags.append(strings)
    
    dp = [float('inf')] * (len(T) + 1)
    dp[0] = 0
    
    for i in range(N):
        for j in range(len(T), -1, -1):
            if dp[j] == float('inf'):
                continue
            for s in bags[i]:
                if T.startswith(s, j):
                    dp[j + len(s)] = min(dp[j + len(s)], dp[j] + 1)
    
    if dp[len(T)] == float('inf'):
        print(-1)
    else:
        print(dp[len(T)])

solve()