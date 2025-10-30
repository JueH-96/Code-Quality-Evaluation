# YOUR CODE HERE
import sys

def calculate_1d_sum(coords):
    """
    Calculates Sum_{0 <= i < j < M} |coords[i] - coords[j]| for a list of coordinates.
    This sum is equivalent to Sum_{k=0}^{M-1} (2k - M + 1) * sorted_coords[k].
    This function implements this formula efficiently.

    Args:
        coords: A list of numbers (integers in this problem).

    Returns:
        The total sum of absolute differences between all pairs of elements in the list.
    """
    M = len(coords)
    # If there are 0 or 1 elements, the sum of differences is 0.
    if M <= 1:
        return 0
    
    # Sort the coordinates. This takes O(M log M) time.
    coords.sort()
    
    total_1d_sum = 0
    # Apply the formula Sum_{k=0}^{M-1} (2k - M + 1) * coords[k]
    # This loop takes O(M) time.
    for k in range(M):
        # The coefficient (2*k - M + 1) represents how many times coords[k] contributes positively
        # minus how many times it contributes negatively in the sum of differences.
        # Specifically, coords[k] is subtracted from k elements larger than it,
        # and (M-1-k) elements smaller than it are subtracted from coords[k].
        # Total contribution of coords[k] is k * coords[k] - (M-1-k) * coords[k] = (k - (M-1-k)) * coords[k] = (2k - M + 1) * coords[k]. No, wait. Let's re-check.
        # Sum = Sum_{j>i} (Z_j - Z_i). For Z_k: it appears as Z_j for k times (j=k). It appears as Z_i for M-1-k times (i=k).
        # So contribution is k * Z_k - (M-1-k) * Z_k = (k - M + 1 + k) Z_k = (2k - M + 1) Z_k. Yes, the formula is correct.
        total_1d_sum += (2 * k - M + 1) * coords[k]
        
    return total_1d_sum

def calculate_subproblem_sum(points_list):
    """
    Calculates the sum of distances for pairs of points within the given list.
    All points in the list are assumed to have the same parity of (X+Y), ensuring reachability
    between any pair according to the problem's distance definition.
    The distance dist(A, B) is defined as the minimum number of rabbit jumps.
    If (X_A+Y_A) and (X_B+Y_B) have the same parity, dist(A, B) = max(|X_A-X_B|, |Y_A-Y_B|).
    This distance is equivalent to 1/2 * Manhattan distance in the transformed (u,v) coordinates,
    where u = X+Y and v = X-Y.
    This function calculates Sum_{i<j} dist(P_i, P_j) for points P_i, P_j in the list.

    Args:
        points_list: A list of tuples, where each tuple is (X, Y) coordinates of a point.

    Returns:
        The total sum of distances between all pairs of points in the list.
    """
    M = len(points_list)
    # If there are 0 or 1 points, the sum of pairwise distances is 0.
    if M <= 1:
        return 0
    
    # Transform coordinates to u=X+Y and v=X-Y.
    u_coords = []
    v_coords = []
    for X, Y in points_list:
        u_coords.append(X + Y)
        v_coords.append(X - Y)
    
    # Calculate the sum of absolute differences for u and v coordinates separately.
    # Sum_{i<j} |u_i - u_j|
    sum_u = calculate_1d_sum(u_coords)
    # Sum_{i<j} |v_i - v_j|
    sum_v = calculate_1d_sum(v_coords)
    
    # The total sum for this subset is Sum_{i<j} max(|Xi-Xj|, |Yi-Yj|)
    # We proved that max(|Xi-Xj|, |Yi-Yj|) = 1/2 * (|ui-uj| + |vi-vj|) when Xi+Yi and Xj+Yj have the same parity.
    # Thus, Total Sum = Sum_{i<j} 1/2 * (|ui-uj| + |vi-vj|)
    #                 = 1/2 * ( Sum_{i<j} |ui-uj| + Sum_{i<j} |vi-vj| )
    #                 = 1/2 * (sum_u + sum_v)
    
    # We also proved that sum_u + sum_v is always even.
    # Therefore, we can use integer division //.
    return (sum_u + sum_v) // 2

# Main part of the program
def solve():
    # Read the number of points
    N = int(sys.stdin.readline())
    
    # Initialize lists to store points based on the parity of X+Y
    points_even_parity = [] # Stores points (X, Y) where X+Y is even
    points_odd_parity = []  # Stores points (X, Y) where X+Y is odd
    
    # Read N points and partition them based on the parity of X+Y
    # The distance between points from different partitions is 0 by definition.
    for _ in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        if (X + Y) % 2 == 0:
            points_even_parity.append((X, Y))
        else:
            points_odd_parity.append((X, Y))

    # Calculate the total sum of distances for pairs of points within the even parity group
    total_sum_even = calculate_subproblem_sum(points_even_parity)
    
    # Calculate the total sum of distances for pairs of points within the odd parity group
    total_sum_odd = calculate_subproblem_sum(points_odd_parity)
    
    # The final answer is the sum of contributions from both groups.
    final_answer = total_sum_even + total_sum_odd
    
    # Print the final answer
    print(final_answer)

# Execute the solve function to run the program
solve()