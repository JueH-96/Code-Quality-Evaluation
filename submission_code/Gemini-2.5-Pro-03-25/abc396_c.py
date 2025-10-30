# YOUR CODE HERE
import sys

def solve():
    # Read N and M, the number of black and white balls respectively.
    N, M = map(int, sys.stdin.readline().split())
    
    # Read the values of the N black balls.
    # Constraints state N >= 1, so B will not be empty.
    B = list(map(int, sys.stdin.readline().split()))
    
    # Read the values of the M white balls.
    # Constraints state M >= 1, so W will not be empty.
    W = list(map(int, sys.stdin.readline().split()))

    # Sort the ball values in descending order.
    # This allows us to easily select the balls with the highest values.
    B.sort(reverse=True)
    W.sort(reverse=True)

    # Compute prefix sums for the sorted black ball values.
    # P_B[k] will store the sum of the values of the top k black balls.
    # P_B[0] is 0, representing the sum when choosing 0 black balls.
    P_B = [0] * (N + 1)
    for i in range(N):
        P_B[i+1] = P_B[i] + B[i]

    # Compute prefix sums for the sorted white ball values.
    # P_W[l] will store the sum of the values of the top l white balls.
    # P_W[0] is 0, representing the sum when choosing 0 white balls.
    P_W = [0] * (M + 1)
    for i in range(M):
        P_W[i+1] = P_W[i] + W[i]

    # Compute suffix maximums of the P_B array.
    # SufMax_B[i] will store the maximum value of P_B[k] for all k such that i <= k <= N.
    # This is useful because if we choose 'l' white balls, we need to choose 'k' black balls where k >= l.
    # To maximize the sum, we want the maximum possible sum of black balls P_B[k] for k in the valid range [l, N].
    SufMax_B = [0] * (N + 1) 
    
    # The base case for the suffix maximum calculation: the maximum for k in [N, N] is just P_B[N].
    SufMax_B[N] = P_B[N]
    
    # Calculate suffix maximums by iterating backwards from N-1 down to 0.
    # SufMax_B[i] is the maximum of P_B[i] itself and the maximum found for the range starting at i+1 (SufMax_B[i+1]).
    for i in range(N - 1, -1, -1):
        SufMax_B[i] = max(P_B[i], SufMax_B[i+1])

    # Initialize the maximum total sum found so far.
    # Since choosing zero balls is always an option (resulting in a sum of 0),
    # the maximum sum must be at least 0.
    max_total_sum = 0 

    # Iterate through all possible numbers of white balls 'l' that we can choose.
    # The number of black balls 'k' must satisfy k >= l and k <= N.
    # This implies that l must be less than or equal to N (since l <= k <= N).
    # Also, we can choose at most M white balls, so l <= M.
    # Combining these, the number of white balls 'l' must be in the range [0, min(N, M)].
    limit_l = min(N, M) 

    for l in range(limit_l + 1):
        # For a fixed number 'l' of white balls, we choose the top 'l' white balls by value.
        # Their total sum is P_W[l].
        # We need to choose 'k' black balls such that k >= l and k <= N.
        # To maximize the total sum P_W[l] + P_B[k], we need to maximize P_B[k] over k in [l, N].
        # The precomputed SufMax_B[l] gives exactly this maximum value: max(P_B[k] for k in [l, N]).
        
        current_sum = P_W[l] + SufMax_B[l]
        
        # Update the overall maximum sum found so far.
        max_total_sum = max(max_total_sum, current_sum)

    # Print the final maximum possible sum.
    print(max_total_sum)

# Execute the solve function to run the logic.
solve()