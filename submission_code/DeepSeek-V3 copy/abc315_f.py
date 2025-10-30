import sys
import math

def main():
    N = int(sys.stdin.readline())
    checkpoints = [tuple(map(int, sys.stdin.readline().split())) for _ in range(N)]
    
    # Initialize DP table
    dp = [float('inf')] * N
    dp[0] = 0
    
    for i in range(N):
        if dp[i] == float('inf'):
            continue
        for j in range(i+1, N):
            # Calculate the Euclidean distance between checkpoint i and j
            dx = checkpoints[j][0] - checkpoints[i][0]
            dy = checkpoints[j][1] - checkpoints[i][1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate the penalty
            skipped = j - i - 1
            if skipped > 0:
                penalty = 2 ** (skipped - 1)
            else:
                penalty = 0
            
            # Update the DP table
            total = dp[i] + distance + penalty
            if total < dp[j]:
                dp[j] = total
    
    print("{0:.20f}".format(dp[-1]))

if __name__ == "__main__":
    main()