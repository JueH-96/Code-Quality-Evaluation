import math

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

def min_s(N, X, Y):
    dp = [float('inf')] * N  # dp[i] represents minimum cost to reach checkpoint i+1
    dp[0] = 0  # cost to reach checkpoint 1 is 0
    
    for i in range(1, N):
        for j in range(i):
            distance = euclidean_distance(X[j], Y[j], X[i], Y[i])
            skipped = i - j - 1
            penalty = 0 if skipped == 0 else 2 ** (skipped - 1)
            dp[i] = min(dp[i], dp[j] + distance + penalty)
    
    return dp[N-1]

def main():
    N = int(input())
    X, Y = [], []
    
    for _ in range(N):
        x, y = map(int, input().split())
        X.append(x)
        Y.append(y)
    
    result = min_s(N, X, Y)
    print(result)

if __name__ == "__main__":
    main()