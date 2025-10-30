import sys

# Use higher precision for floating point calculations and comparisons
# Use 1e-12 as a small tolerance
EPS = 1e-12

def solve():
    N = int(sys.stdin.readline())
    buildings = []
    for _ in range(N):
        X, H = map(int, sys.stdin.readline().split())
        buildings.append((float(X), float(H))) # Use float from the start

    # Upper convex hull of points (X_j, H_j) for j < i
    # Stores points (X, H)
    hull = []
    max_h_crit = -float('inf') # Initialize with negative infinity

    for i in range(N):
        Xi, Hi = buildings[i]

        if i > 0:
            # Find P_j = (X_j, H_j) in hull that maximizes (Hi * X_j - H_j * Xi) / (X_j - Xi)
            # This is equivalent to minimizing slope (Hi - H_j) / (Xi - X_j)
            # The points (X_j, H_j) are in hull. P_i = (Xi, Hi).
            # Minimize slope (Hi - hy) / (Xi - hx) for (hx, hy) in hull.

            best_j_idx = 0 # Default index

            # Use ternary search to find the index k that minimizes slope(hull[k], P_i)
            # The sequence of slopes slope(hull[k], P_i) as k increases is unimodal (convex when viewed from above)
            low_bs = 0
            high_bs = len(hull) - 1

            # Need at least one point in hull to calculate critical height
            if len(hull) >= 1:
                # Ternary search requires at least 2 points in the search interval for comparison (high_bs - low_bs >= 1)
                # The loop continues while there are at least 3 points (high_bs - low_bs >= 2)
                while high_bs - low_bs >= 2:
                    mid1 = low_bs + (high_bs - low_bs) // 3
                    mid2 = high_bs - (high_bs - low_bs) // 3

                    # Ensure mid1 and mid2 are distinct
                    # This check is primarily for safety/clarity, should be handled by loop condition >= 2
                    if mid1 == mid2:
                         mid2 += 1 # Should be safe as high_bs >= mid1 + 1

                    slope1 = (Hi - hull[mid1][1]) / (Xi - hull[mid1][0])
                    slope2 = (Hi - hull[mid2][1]) / (Xi - hull[mid2][0])

                    # If slope1 > slope2, the minimum must be at mid1 or to the right of mid1
                    # If slope1 <= slope2, the minimum must be at mid2 or to the left of mid2
                    if slope1 > slope2 + EPS:
                        low_bs = mid1 + 1
                    else: # slope1 <= slope2
                        high_bs = mid2

                # After the loop, the minimum slope is in the remaining interval [low_bs, high_bs]
                # This interval has size 1 or 2. Check all points in this interval.
                best_j_idx = low_bs
                # Need at least one point in hull, checked by `if len(hull) >= 1`
                # Initialize min_slope with the slope at the first point in the remaining interval
                min_slope = (Hi - hull[low_bs][1]) / (Xi - hull[low_bs][0])
                
                # Check the rest of the points in the small interval
                for k in range(low_bs + 1, high_bs + 1):
                    current_slope = (Hi - hull[k][1]) / (Xi - hull[k][0])
                    if current_slope < min_slope - EPS:
                        min_slope = current_slope
                        best_j_idx = k

                X_j_star, H_j_star = hull[best_j_idx]
                
                # Calculate the critical height h for the line through (X_j_star, H_j_star) and (Xi, Hi) intersecting y-axis
                # h = (Hi * X_j_star - H_j_star * Xi) / (X_j_star - Xi)
                # X_j_star < Xi, so X_j_star - Xi is negative and non-zero
                h_ij_crit = (Hi * X_j_star - H_j_star * Xi) / (X_j_star - Xi)
                max_h_crit = max(max_h_crit, h_ij_crit)


        # Add current building P_i to the upper convex hull
        # Maintain the upper convex hull property: remove points that create a non-clockwise turn
        # Use cross product check: (B-A) x (C-B) <= 0 for points A, B, C (A=hull[-2], B=hull[-1], C=P_i)
        # Cross product = (x2-x1)(y3-y2) - (y2-y1)(x3-x2)
        while len(hull) >= 2:
            P_last = hull[-1]
            P_second_last = hull[-2]
            x1, y1 = P_second_last
            x2, y2 = P_last
            x3, y3 = Xi, Hi
            
            # Calculate cross product using double
            cross_product = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
            
            # If cross product is > 0 (counter-clockwise turn), remove the last point (P_last)
            # Use tolerance for floating point comparison
            if cross_product > EPS:
                hull.pop()
            else:
                break # Clockwise or straight turn, hull property maintained

        # Add the current building P_i to the hull
        hull.append((Xi, Hi))

    # Final check: if max_h_crit < 0, all buildings are visible from h=0
    # Use tolerance when checking against zero
    if max_h_crit < -EPS:
        print(-1)
    else:
        print(f'{max_h_crit:.18f}') # Print with required precision

solve()