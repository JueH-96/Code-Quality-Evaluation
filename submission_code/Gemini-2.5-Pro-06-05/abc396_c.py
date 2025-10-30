# YOUR CODE HERE
import sys

def solve():
    """
    Solves the ball selection problem by iterating through the number of black balls
    and finding the optimal number of white balls using precomputed prefix sums.
    """
    # Read input from stdin for efficiency
    try:
        readline = sys.stdin.readline
        N, M = map(int, readline().split())
        B = list(map(int, readline().split()))
        W = list(map(int, readline().split()))
    except (IOError, ValueError):
        # Handle cases with malformed or empty input lines
        return

    # Sort ball values in descending order to ensure we always pick the most valuable ones.
    B.sort(reverse=True)
    W.sort(reverse=True)

    # pB[k] will store the sum of the top k black balls.
    pB = [0] * (N + 1)
    for i in range(N):
        pB[i + 1] = pB[i] + B[i]

    # pW[k] will store the sum of the top k white balls.
    pW = [0] * (M + 1)
    for i in range(M):
        pW[i + 1] = pW[i] + W[i]

    # max_pW[k] stores the maximum sum achievable by choosing j white balls, where 0 <= j <= k.
    # This is equivalent to max(pW[0], pW[1], ..., pW[k]).
    max_pW = [0] * (M + 1)
    for i in range(1, M + 1):
        max_pW[i] = max(max_pW[i - 1], pW[i])
        
    # The answer is at least 0 (by choosing no balls).
    ans = 0

    # Iterate through all possible numbers of black balls to choose (k_b).
    for k_b in range(N + 1):
        # For a fixed k_b, the number of white balls k_w must satisfy k_w <= k_b and k_w <= M.
        # To maximize the total sum, we need to pick k_w in [0, min(k_b, M)]
        # that maximizes the sum of white ball values. This is precomputed in max_pW.
        k_w_limit = min(k_b, M)
        
        # The sum from choosing the top k_b black balls + the best possible sum from white balls.
        current_sum = pB[k_b] + max_pW[k_w_limit]
        
        # Update the overall maximum sum found so far.
        ans = max(ans, current_sum)
            
    # Print the final answer.
    print(ans)

solve()