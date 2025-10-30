import sys
import math

def minimum_distance_with_penalty(N, checkpoints):
    # Initialize a list to store the minimum distance to reach each checkpoint
    dp = [float('inf')] * N
    dp[0] = 0  # Starting point, no distance traveled

    # Iterate through each checkpoint
    for i in range(N):
        for j in range(i + 1, N):
            # Calculate the Euclidean distance from checkpoint i to checkpoint j
            distance = math.sqrt((checkpoints[j][0] - checkpoints[i][0]) ** 2 + 
                                 (checkpoints[j][1] - checkpoints[i][1]) ** 2)
            # Update the distance to reach checkpoint j
            dp[j] = min(dp[j], dp[i] + distance)

    # Now we need to consider the penalties
    min_s = float('inf')
    
    # Check all possible skips
    for skips in range(N - 1):  # skips can be from 0 to N-2
        penalty = 2 ** skips if skips > 0 else 0
        # The total distance with the current number of skips
        total_distance = dp[N - 1] + penalty
        min_s = min(min_s, total_distance)

    return min_s

def main():
    input = sys.stdin.read
    data = input().splitlines()
    
    N = int(data[0])
    checkpoints = [tuple(map(int, line.split())) for line in data[1:N + 1]]
    
    result = minimum_distance_with_penalty(N, checkpoints)
    print(f"{result:.10f}")

if __name__ == "__main__":
    main()