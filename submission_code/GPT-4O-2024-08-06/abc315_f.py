import sys
import math

def euclidean_distance(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    checkpoints = []
    
    index = 1
    for _ in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        checkpoints.append((x, y))
        index += 2
    
    # Initialize dp array
    dp = [float('inf')] * N
    dp[0] = 0  # Starting point, no cost
    
    # Fill dp array
    for i in range(1, N):
        for j in range(i):
            distance = euclidean_distance(checkpoints[j][0], checkpoints[j][1], checkpoints[i][0], checkpoints[i][1])
            penalty = 2 ** (i - j - 1) if i - j - 1 > 0 else 0
            dp[i] = min(dp[i], dp[j] + distance + penalty)
    
    # Output the minimum cost to reach the last checkpoint
    print(dp[N-1])