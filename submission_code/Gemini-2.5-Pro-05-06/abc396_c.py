import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())
    
    B_values = list(map(int, sys.stdin.readline().split()))
    W_values = list(map(int, sys.stdin.readline().split()))

    B_values.sort(reverse=True)
    W_values.sort(reverse=True)

    # Prefix sums for B_values
    # P_B[k] = sum of k largest black ball values
    # P_B has N+1 elements, P_B[0] = 0
    P_B = [0] * (N + 1)
    for i in range(N):
        P_B[i+1] = P_B[i] + B_values[i]

    # Prefix sums for W_values
    # P_W[k] = sum of k largest white ball values
    # P_W has M+1 elements, P_W[0] = 0
    P_W = [0] * (M + 1)
    for i in range(M):
        P_W[i+1] = P_W[i] + W_values[i]
    
    # Suffix maximums of P_B array
    # SufMaxP_B[k] stores max(P_B[j] for j from k to N inclusive)
    # SufMaxP_B has N+1 elements.
    SufMaxP_B = list(P_B) # Initialize with values from P_B
                          # Base case: SufMaxP_B[N] is P_B[N] which is correct.
                          # If N=0, P_B=[0], SufMaxP_B=[0]. Loop below doesn't run. Correct.
    for i in range(N - 1, -1, -1): # Iterate from N-1 down to 0
        SufMaxP_B[i] = max(SufMaxP_B[i], SufMaxP_B[i+1])
            
    max_overall_sum = 0 # Minimum possible sum is 0 (by choosing no balls)

    # Iterate over number of white balls chosen, k_W.
    # k_W can range from 0 to M.
    # Number of black balls chosen, k_B, must satisfy k_B >= k_W.
    # Also, k_B <= N (at most N black balls available).
    # So, k_W <= k_B <= N implies k_W <= N.
    # Thus, k_W effectively ranges from 0 to min(M, N).
    limit_k_W = min(M, N)

    for k_W in range(limit_k_W + 1): # k_W from 0 to limit_k_W inclusive
        current_sum_W = P_W[k_W]
        
        # We need to choose k_B black balls, where k_W <= k_B <= N.
        # The sum from these black balls is P_B[k_B].
        # We want to maximize P_B[k_B] for k_B in [k_W, N]. This value is SufMaxP_B[k_W].
        max_sum_B_for_this_k_W = SufMaxP_B[k_W]
        
        current_total_sum = current_sum_W + max_sum_B_for_this_k_W
        if current_total_sum > max_overall_sum:
            max_overall_sum = current_total_sum
            
    print(max_overall_sum)

if __name__ == '__main__':
    solve()