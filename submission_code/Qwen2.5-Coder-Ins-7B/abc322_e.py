import sys
from typing import List

def main():
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    N = int(data[index])
    K = int(data[index + 1])
    P = int(data[index + 2])
    index += 3
    
    plans = []
    for _ in range(N):
        C = int(data[index])
        A = list(map(int, data[index + 1:index + K + 1]))
        plans.append((C, A))
        index += K + 1
    
    dp = [[float('inf')] * (P + 1) for _ in range(K + 1)]
    dp[0][0] = 0
    
    for i in range(1, K + 1):
        for j in range(P + 1):
            dp[i][j] = dp[i - 1][j]
            if j >= A[i - 1]:
                dp[i][j] = min(dp[i][j], dp[i - 1][j - A[i - 1]] + C)
    
    if dp[K][P] == float('inf'):
        print(-1)
    else:
        print(dp[K][P])

if __name__ == "__main__":
    main()