import math

# Function to calculate Euclidean distance between two points
def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Function to calculate the penalty based on the number of skipped checkpoints C
def calculate_penalty(C):
    if C == 0:
        return 0.0
    # Penalty is 2^(C-1) for C > 0
    return 2**(C - 1)

def solve():
    N = int(input())
    
    points = []
    for _ in range(N):
        x, y = map(int, input().split())
        points.append((x, y))

    # K_LIMIT is the maximum number of skipped checkpoints we consider.
    # If the number of skipped checkpoints C is too high, the penalty 2^(C-1)
    # becomes prohibitively large, making such paths suboptimal.
    # For N=10^4, max distance could be around 1.4 * 10^8.
    # 2^27 is approx 1.34 * 10^8. 2^28 is approx 2.68 * 10^8.
    # So, paths with more than ~28 skipped checkpoints are likely not optimal.
    # K_LIMIT = 30 provides a safe margin.
    K_LIMIT = 30 

    # dp[i][k] stores the minimum total distance to reach checkpoint `i` (0-indexed),
    # having skipped exactly `k` checkpoints in total among those from 2 to `i`.
    # Initialize with a very large value (infinity).
    dp = [[float('inf')] * (K_LIMIT + 1) for _ in range(N)]

    # Base case: Start at checkpoint 0 (which is checkpoint 1), 0 skips. Distance is 0.
    dp[0][0] = 0.0

    # Iterate through all possible current checkpoints `i` (from 0 to N-1)
    for i in range(N):
        # Iterate through all possible counts of total skipped checkpoints `k_total` up to `i`
        for k_total in range(K_LIMIT + 1):
            # If we haven't found a way to reach `i` with `k_total` skips, skip this state
            if dp[i][k_total] == float('inf'):
                continue

            # From checkpoint `i`, try to jump to a future checkpoint `j`.
            # `num_skipped_in_jump` is the count of checkpoints (excluding i and j)
            # that are skipped between checkpoint `i` and checkpoint `j`.
            # This loop ensures that the `new_k_total` (k_total + num_skipped_in_jump)
            # does not exceed `K_LIMIT`.
            for num_skipped_in_jump in range(K_LIMIT - k_total + 1):
                j = i + num_skipped_in_jump + 1 # Calculate index of the next checkpoint `j`
                
                # If `j` is beyond the last checkpoint (N-1), we cannot make this jump.
                if j >= N:
                    break # No more valid next checkpoints from `i` for this `num_skipped_in_jump`

                new_k_total = k_total + num_skipped_in_jump
                
                # Calculate the Euclidean distance for this segment
                dist_ij = euclidean_distance(points[i], points[j])
                
                # Update the minimum distance to reach checkpoint `j` with `new_k_total` skips
                dp[j][new_k_total] = min(dp[j][new_k_total], dp[i][k_total] + dist_ij)

    # After filling the DP table, find the minimum total cost for reaching the final checkpoint (N-1).
    # The total cost includes the accumulated distance and the penalty for the total skips.
    min_s = float('inf')
    for k_total in range(K_LIMIT + 1):
        if dp[N-1][k_total] != float('inf'):
            current_s = dp[N-1][k_total] + calculate_penalty(k_total)
            min_s = min(min_s, current_s)

    # Print the result with required precision (10 decimal places for safety)
    print(f"{min_s:.10f}")

# Execute the solution
solve()