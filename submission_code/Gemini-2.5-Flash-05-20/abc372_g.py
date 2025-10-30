import sys
from collections import deque
from functools import cmp_to_key

# Function to compute sum_{i=0}^{n-1} floor((A*i + B) / C)
# This is a standard algorithm for sum of floors.
# Parameters A, B can be negative, C must be positive.
# n is the number of terms. If n=0, sum is 0.
def floor_sum(n, A, B, C):
    res = 0
    if n == 0:
        return 0

    # Step 1: Handle cases where A < 0 by factoring out quotient
    # A*i + B = (q_A * C + (A % C)) * i + B
    # floor((A*i + B) / C) = q_A * i + floor(((A % C)*i + B) / C)
    # Sum(q_A * i) = q_A * sum(i) = q_A * n*(n-1)/2
    if A < 0:
        q_A = A // C
        res += q_A * (n - 1) * n // 2
        A = A % C # Python's % gives result in [0, C-1] if C>0.

    # Step 2: Handle cases where B < 0 by factoring out quotient
    # A*i + B = A*i + (q_B * C + (B % C))
    # floor((A*i + B) / C) = q_B + floor((A*i + (B % C)) / C)
    # Sum(q_B) = q_B * n
    if B < 0:
        q_B = B // C
        res += q_B * n
        B = B % C # Python's % gives result in [0, C-1] if C>0.
            
    # Now A and B are guaranteed to be non-negative because Python's % operator
    # for `X % Y` where `Y > 0` returns a result `Z` such that `0 <= Z < Y`.
    # However, A or B might still be >= C from initial large positive values,
    # so we need to normalize them. This logic is also covered below.

    # Step 3: Handle cases where A >= C or B >= C (normalize A and B)
    if A >= C:
        q_A = A // C
        res += q_A * (n - 1) * n // 2
        A = A % C

    if B >= C:
        q_B = B // C
        res += q_B * n
        B = B % C

    # Step 4: Base cases for recursion
    # If A is 0, the terms are constant (B/C), sum is n * (B/C).
    if A == 0:
        return res + (B // C) * n

    # Step 5: Recursive step (Euclidean algorithm style)
    # This identity is derived from counting lattice points.
    # sum_{i=0}^{n-1} floor((A*i + B) / C)
    #   = n * floor((A*(n-1) + B) / C) - sum_{j=0}^{floor((A*(n-1)+B)/C)-1} floor((C*j + C - B - 1) / A)
    # Let m_val = floor((A*(n-1) + B) / C), which is the maximum value of floor((A*i+B)/C)
    # for i in [0, n-1].
    m_val = (A * (n - 1) + B) // C
    
    # If m_val is 0, all terms were 0 after normalization, no further sum.
    if m_val == 0:
        return res

    # Recursive call
    res += (n * m_val - floor_sum(m_val, C, C - B - 1, A))
    return res


def solve():
    N = int(sys.stdin.readline())
    constraints = []
    
    # Determine the maximum x value for which a positive y can exist.
    # A_i * x + B_i * y < C_i
    # Since y >= 1, we must have A_i * x + B_i * 1 < C_i
    # A_i * x < C_i - B_i
    # x < (C_i - B_i) / A_i
    # So, x <= floor((C_i - B_i - 1) / A_i) for each i.
    # The overall max_overall_x is the minimum of these individual limits.
    max_overall_x = float('inf') # Use float('inf') for initial large value

    for _ in range(N):
        A, B, C = map(int, sys.stdin.readline().split())
        constraints.append((A, B, C))
        
        # Calculate x_limit for current constraint (A,B,C)
        # If (C - B - 1) is negative, (C - B - 1) // A will be negative or 0.
        # This correctly implies no positive x can satisfy this for y >= 1.
        x_limit_for_this_i = (C - B - 1) // A
        max_overall_x = min(max_overall_x, x_limit_for_this_i)
            
    # If the combined max_overall_x is less than 1, no positive x exists.
    if max_overall_x < 1:
        print(0)
        return

    # Sort constraints to build the lower envelope (Convex Hull Trick)
    # Lines are y = (C_i - 1 - A_i * x) / B_i. Slope is -A_i / B_i.
    # We want to sort by slope ascending (-A_i/B_i decreasing, i.e., A_i/B_i increasing).
    # Comparison function for (A1, B1, C1) and (A2, B2, C2):
    # Compare A1/B1 and A2/B2, which is A1*B2 vs A2*B1 (to avoid floats).
    # If slopes are equal, pick the one with a smaller y-intercept (C-1)/B.
    # (C1-1)/B1 vs (C2-1)/B2, which is (C1-1)*B2 vs (C2-1)*B1.
    def compare_lines(p1, p2):
        # p1=(A1,B1,C1), p2=(A2,B2,C2)
        slope_cmp = p1[0] * p2[1] - p2[0] * p1[1]
        if slope_cmp != 0:
            return slope_cmp
        # Slopes are equal, compare y-intercepts. Smaller intercept wins.
        intercept_cmp = (p1[2] - 1) * p2[1] - (p2[2] - 1) * p1[1]
        return intercept_cmp

    constraints.sort(key=cmp_to_key(compare_lines))
    
    # Build lower envelope using a stack (deque for efficient operations)
    hull = deque() # Stores (A, B, C) of lines forming the lower envelope
    
    # Function to calculate x-coordinate of intersection of two lines
    # Line1: (A1, B1, C1), Line2: (A2, B2, C2)
    # y = (C-1-Ax)/B
    # Returns (numerator, denominator) for x_intersect to handle precision
    def get_intersect_x(l1, l2):
        A1, B1, C1 = l1
        A2, B2, C2 = l2
        num = (C2 - 1) * B1 - (C1 - 1) * B2
        den = A2 * B1 - A1 * B2
        # den should be positive (A2*B1 > A1*B2 because A2/B2 > A1/B1)
        return num, den

    for line in constraints:
        # Prune redundant lines from the hull.
        # L_prev1 = hull[-1], L_prev2 = hull[-2]
        # L_prev1 is redundant if x_intersect(L_prev2, L_prev1) >= x_intersect(L_prev1, L_new_line)
        while len(hull) >= 2:
            L_prev1 = hull[-1]
            L_prev2 = hull[-2]
            
            num_12, den_12 = get_intersect_x(L_prev2, L_prev1)
            num_1N, den_1N = get_intersect_x(L_prev1, line)
            
            # Compare num_12/den_12 >= num_1N/den_1N
            # Since denominators are positive, cross-multiply: num_12 * den_1N >= num_1N * den_12
            if num_12 * den_1N >= num_1N * den_12:
                hull.pop() # L_prev1 is redundant
            else:
                break
        hull.append(line)

    # Sum up the counts for each segment defined by the hull lines
    total_pairs_count = 0
    current_x = 1 # Start checking from x=1
    
    for i in range(len(hull)):
        A_k, B_k, C_k = hull[i]
        
        # Determine the effective end x for the current line's segment.
        # This is capped by max_overall_x and the intersection with the next hull line.
        segment_end_x_by_next_line = max_overall_x 
        if i < len(hull) - 1:
            L_next = hull[i+1]
            num_intersect, den_intersect = get_intersect_x(hull[i], L_next)
            
            # The x-coordinate where the current line intersects the next.
            # Current line is active up to floor(intersect_x).
            x_intersect_val_floor = num_intersect // den_intersect
            segment_end_x_by_next_line = x_intersect_val_floor
        
        # Additional constraint: y must be >= 1 for the current line.
        # (C_k - 1 - A_k * x) / B_k >= 1
        # C_k - 1 - A_k * x >= B_k
        # C_k - 1 - B_k >= A_k * x
        # x <= (C_k - 1 - B_k) / A_k
        y_positive_x_limit = (C_k - 1 - B_k) // A_k
        
        # The actual end of the current segment is the minimum of all upper bounds.
        segment_effective_end_x = min(max_overall_x, segment_end_x_by_next_line, y_positive_x_limit)
        
        # Calculate the range for summation for this segment
        start_x_for_segment = current_x
        end_x_for_segment = segment_effective_end_x
        
        # If the start x for this segment is already beyond its end x, skip.
        if start_x_for_segment > end_x_for_segment:
            current_x = end_x_for_segment + 1 # Advance current_x past this segment's valid range
            continue
            
        # Sum floor((C_k-1 - A_k * x) / B_k) for x from start_x_for_segment to end_x_for_segment
        # This is equivalent to sum_{i=0}^{num_terms-1} floor(( (C_k-1 - A_k*start_x_for_segment) - A_k*i ) / B_k)
        num_terms = end_x_for_segment - start_x_for_segment + 1
        
        # Parameters for floor_sum function:
        # n = num_terms
        # A_param = -A_k (coefficient of i)
        # B_param = C_k - 1 - A_k * start_x_for_segment (constant term)
        # C_param = B_k (denominator)
        
        segment_sum = floor_sum(num_terms, -A_k, C_k - 1 - A_k * start_x_for_segment, B_k)
        total_pairs_count += segment_sum
        
        # Move to the next x value for the subsequent segment
        current_x = segment_end_x_by_next_line + 1 # This logic must be careful, current_x should go to floor(intersection)+1
                                                  # segment_end_x_by_next_line is the upper limit for this line, so next valid x starts after it.
        
        # If current_x exceeds max_overall_x, we are done
        if current_x > max_overall_x:
            break
            
    print(total_pairs_count)

# Read number of test cases
T = int(sys.stdin.readline())
for _ in range(T):
    solve()