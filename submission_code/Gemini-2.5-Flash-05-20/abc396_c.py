# YOUR CODE HERE
import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    B = list(map(int, sys.stdin.readline().split()))
    W = list(map(int, sys.stdin.readline().split()))

    # 1. Sort B in descending order to easily pick largest values
    B.sort(reverse=True)
    # 2. Sort W in descending order to easily pick largest values
    W.sort(reverse=True)

    # 3. Compute prefix sums for B
    # prefix_B[i] stores sum(B[0]...B[i-1])
    prefix_B = [0] * (N + 1)
    for i in range(N):
        prefix_B[i+1] = prefix_B[i] + B[i]

    # 4. Compute prefix sums for W
    # prefix_W[i] stores sum(W[0]...W[i-1])
    prefix_W = [0] * (M + 1)
    for i in range(M):
        prefix_W[i+1] = prefix_W[i] + W[i]

    # 5. Compute suffix sums of positive values for B
    # B_suffix_pos_sum[i] stores sum of B_j for j >= i where B_j > 0
    B_suffix_pos_sum = [0] * (N + 1)
    current_positive_sum = 0
    # Iterate from N-1 down to 0
    for i in range(N - 1, -1, -1):
        if B[i] > 0:
            current_positive_sum += B[i]
        B_suffix_pos_sum[i] = current_positive_sum
    
    # 6. Initialize max_total_sum
    # It's possible to choose no balls, resulting in a sum of 0.
    # Consider the case where we choose 0 white balls (k=0).
    # In this scenario, N_w = 0, and we only need to pick black balls.
    # To maximize sum, we pick all positive black balls from B[0...]
    # This sum is B_suffix_pos_sum[0].
    max_total_sum = max(0, B_suffix_pos_sum[0])

    # 7. Iterate through possible counts of chosen white balls (k)
    # k can range from 1 to min(N, M).
    # For a fixed k:
    # - We choose k largest white balls (W[0]...W[k-1])
    # - We choose k largest black balls (B[0]...B[k-1]) to satisfy N_b >= N_w initially.
    # - We then add any additional positive black balls from B[k...]
    for k in range(1, min(N, M) + 1):
        # Sum of k largest white balls + k largest black balls
        current_sum = prefix_W[k] + prefix_B[k]
        
        # Add sum of positive black balls from index k onwards
        current_sum += B_suffix_pos_sum[k] 
        
        max_total_sum = max(max_total_sum, current_sum)

    print(max_total_sum)

solve()
# END YOUR CODE