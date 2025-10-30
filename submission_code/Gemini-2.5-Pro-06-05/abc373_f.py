# YOUR CODE HERE
import sys
from collections import deque

def main():
    """
    Solves the problem using dynamic programming with Convex Hull Trick optimization.
    """
    try:
        input = sys.stdin.readline
    except AttributeError:
        # For local testing if running from a file
        pass

    N, W = map(int, input().split())
    items = [tuple(map(int, input().split())) for _ in range(N)]
    
    # Use a large negative number for unreachable states. Happiness can be negative.
    neg_inf = -10**18
    
    dp = [neg_inf] * (W + 1)
    dp[0] = 0
    
    for w_i, v_i in items:
        # `dp` represents the state after the previous item.
        # `next_dp` will be the state after processing the current item.
        next_dp = dp[:]
        
        for r in range(w_i):
            # `dq` stores the convex hull of points (j, A_j) for a fixed remainder `r`.
            dq = deque()
            
            # `m` is the number of items of type `i` being considered.
            max_m = (W - r) // w_i
            
            for m in range(max_m + 1):
                # We are calculating the value for next_dp[r + m * w_i].
                # This corresponds to `g(m)` in the CHT formulation.

                # Step 1: Add point for j=m to the convex hull.
                # The point is derived from the previous DP state `dp`.
                weight_j = r + m * w_i
                f_j = dp[weight_j]
                
                if f_j > neg_inf:
                    # A_j = f(j) - j*v_i - j^2
                    A_j = f_j - m * v_i - m * m
                    p_new = (m, A_j)
                    
                    # Maintain the upper convex hull (concave chain) by removing points
                    # that are made redundant by the new point.
                    while len(dq) >= 2:
                        p1 = dq[-1]
                        p2 = dq[-2]
                        # Cross-product condition to check for concavity.
                        # slope(p2, p1) >= slope(p1, p_new)
                        if (p1[1] - p2[1]) * (p_new[0] - p1[0]) <= (p_new[1] - p1[1]) * (p1[0] - p2[0]):
                            dq.pop()
                        else:
                            break
                    dq.append(p_new)

                # Step 2: Query the hull to find the best value for the current `m`.
                if dq:
                    # Query with slope `2*m`.
                    # Since slopes are increasing, the optimal point on the hull will
                    # move forward. We can remove non-optimal points from the front.
                    while len(dq) >= 2:
                        p1 = dq[0]
                        p2 = dq[1]
                        # Compare objective function value for p1 and p2.
                        # obj(j) = A_j + 2*m*j
                        if p2[1] + 2 * m * p2[0] >= p1[1] + 2 * m * p1[0]:
                            dq.popleft()
                        else:
                            break
                    
                    j_opt, A_j_opt = dq[0]
                    # This is the max_{j} part of the CHT formula
                    max_part = A_j_opt + 2 * m * j_opt
                    
                    # Calculate the full happiness value `g(m)`
                    new_val = (m * v_i - m * m) + max_part
                    
                    current_weight = r + m * w_i
                    next_dp[current_weight] = max(next_dp[current_weight], new_val)
        
        # Update the DP table for the next item
        dp = next_dp
    
    # The maximum happiness could be 0 (by taking nothing).
    final_answer = 0
    for val in dp:
        if val > neg_inf:
            final_answer = max(final_answer, val)
            
    print(final_answer)

if __name__ == "__main__":
    main()