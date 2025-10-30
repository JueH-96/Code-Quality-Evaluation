# YOUR CODE HERE
import sys
from collections import deque
import math

# Using sys.stdin.readline for faster input
input = sys.stdin.readline

# Function to compare slopes slope(B, C) < slope(A, C) using integer arithmetic
# Checks if the slope of the line segment connecting B and C is less than the slope
# of the line segment connecting A and C.
# Uses integer arithmetic with Python's arbitrary precision integers to avoid float precision issues during comparison.
def compare_slopes_lt(A, B, C):
    """
    Compares the slopes of line segments AC and BC.
    Returns True if slope(BC) < slope(AC), False otherwise.
    Points A, B, C are tuples (X, H).
    Assumes X coordinates are strictly increasing: C[0] > B[0] > A[0].
    Calculation uses cross multiplication: (C[1] - B[1]) * (C[0] - A[0]) < (C[1] - A[1]) * (C[0] - B[0])
    """
    # Calculate the terms for cross multiplication using Python's arbitrary precision integers.
    val1 = (C[1] - B[1]) * (C[0] - A[0])
    val2 = (C[1] - A[1]) * (C[0] - B[0])
    # The comparison holds if val1 < val2.
    return val1 < val2

# Function to check cross product for hull maintenance: non-left turn A, B, C?
# Checks if the sequence of points A, B, C forms a non-left turn (i.e., C lies to the right
# of or is collinear with the directed line segment AB).
# Used to maintain the lower convex hull property. If true, point B is redundant and should be removed.
def cross_product_non_left(A, B, C):
    """
    Checks the orientation of point C relative to the directed line segment AB.
    Returns True if A, B, C make a non-left turn (cross product <= 0), False otherwise (left turn, cross product > 0).
    Points A, B, C are tuples (X, H).
    Calculation uses the z-component of the cross product of vectors AB and BC:
    (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])
    """
    # Calculate the cross product using Python's arbitrary precision integers.
    val = (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])
    # A non-left turn corresponds to cross product <= 0.
    return val <= 0

def solve():
    N = int(input())
    
    # If there's only one building, it's always visible from (0,0). Problem asks for max height
    # where NOT all buildings are visible. With N=1, all buildings are always visible.
    # So if N=1, visibility from (0,0) is guaranteed. Thus, report -1.
    if N == 1:
        print("-1")
        return

    buildings = []
    for _ in range(N):
        buildings.append(tuple(map(int, input().split())))

    # `hull` is a deque storing vertices (tuples (X, H)) of the lower convex hull
    # of the top points of buildings processed so far.
    hull = deque()
    
    # `max_h` stores the maximum height found so far at coordinate 0 from which
    # at least one building is not visible. Initialize to negative infinity.
    max_h = -math.inf 
    
    # Process the first building T_0 = (X_0, H_0)
    T0 = buildings[0]
    hull.append(T0)
    
    # `opt_idx` stores the index in `hull` of the point T_j that gave the minimum slope
    # connecting to the previous point T_{i-1}. This is used as a starting point
    # for the search for the optimal point for T_i, exploiting the property that
    # the optimal index tends to move monotonically along the hull.
    opt_idx = 0 

    # Iterate through buildings from the second one (index 1) up to N-1
    for i in range(1, N):
        T_i = buildings[i] # Current building's top point (X_i, H_i)
        
        p = len(hull) # Current number of points in the hull
        
        # Ensure opt_idx is a valid index within the current hull bounds `[0, p-1]`.
        # This adjustment is crucial because points might have been removed from the end
        # of the hull in the previous iteration's update step, potentially invalidating `opt_idx`.
        opt_idx = min(opt_idx, p - 1) 

        # Query step: Find the point T_j on the hull that minimizes the slope of the line segment T_j T_i.
        # We utilize the property that the function slope(hull[k], T_i) is convex-like with respect to k.
        # Since T_i moves rightward (X_i increases), the index `k` minimizing the slope tends to increase.
        # We start searching from the previously found optimal index `opt_idx` and move rightwards.
        while opt_idx + 1 < p and compare_slopes_lt(hull[opt_idx], hull[opt_idx + 1], T_i):
             # If the slope decreases by moving to the next point on the hull (hull[opt_idx+1]),
             # update opt_idx to point to this better point (index opt_idx + 1).
             opt_idx += 1

        # After the loop, hull[opt_idx] corresponds to the point T_j that minimizes slope(T_j, T_i)
        T_j = hull[opt_idx]
        
        # Calculate the candidate maximum height h_i. This height is the y-intercept B_ij
        # of the line passing through T_j=(X_j, H_j) and T_i=(X_i, H_i).
        # The formula for the y-intercept is B_ij = (H_j * X_i - H_i * X_j) / (X_i - X_j).
        # We compute the numerator and denominator using integer arithmetic first for maximum precision,
        # then perform float division.
        numerator = T_j[1] * T_i[0] - T_i[1] * T_j[0]
        denominator = T_i[0] - T_j[0]
        
        # The denominator is guaranteed to be positive since X_i > X_j.
        # Perform float division to get the height h_i.
        h_i = float(numerator) / float(denominator) 
        
        # Update the overall maximum height found so far.
        max_h = max(max_h, h_i)

        # Update hull step: Add T_i to the hull while maintaining the lower convex hull property.
        # Points that violate the convexity property (cause a non-left turn) are removed from the end of the deque.
        p = len(hull) # Refresh hull length before modifications
        while p >= 2:
            # Consider the last segment on the hull (A, B) where A=hull[p-2], B=hull[p-1],
            # and check its turn direction with the new point C=T_i.
            A = hull[p-2]
            B = hull[p-1]
            # If A, B, T_i form a non-left turn (C is right of or collinear with directed segment AB),
            # then B is "hidden" below the segment AC (or AC contains B) and is thus
            # not part of the lower hull when T_i is included. Remove B.
            if cross_product_non_left(A, B, T_i):
                 hull.pop() # Remove B from the end
                 p -= 1    # Decrease hull size counter
                 # If the removed point B was the one pointed to by opt_idx, or potentially a point
                 # before it that opt_idx should now point to, we must ensure opt_idx remains valid.
                 # Setting opt_idx = min(opt_idx, p - 1) correctly adjusts it to stay within bounds [0, p-1].
                 opt_idx = min(opt_idx, p - 1)
            else:
                # If A, B, T_i form a left turn, the convexity property is maintained.
                # Stop removing points. B remains on the hull.
                break 
        
        # Add the current point T_i to the end of the hull.
        hull.append(T_i)

    # After processing all buildings, decide the final output based on the calculated max_h.
    # If max_h is negative, it implies that from height 0, all buildings were visible.
    # (Because any h < 0 is not allowed, and for h=0, visibility holds if max_h < 0).
    # The problem asks to report -1 in this specific case.
    if max_h < 0:
         print("-1")
    else:
        # Otherwise, max_h is the maximum non-negative height from which not all buildings are visible.
        # Print this value formatted to the required high precision.
        print(f"{max_h:.18f}")

# Execute the solve function when the script runs.
solve()