# YOUR CODE HERE
import sys
import bisect
from collections import Counter

def solve():
    # Read N and D from input
    N, D = map(int, sys.stdin.readline().split())
    
    # Handle edge case N=0, though constraints state N >= 1
    if N == 0: 
        print(0)
        return

    # Read N points coordinates
    X_coords = []
    Y_coords = []
    for _ in range(N):
        xi, yi = map(int, sys.stdin.readline().split())
        X_coords.append(xi)
        Y_coords.append(yi)

    # Sort coordinates lists. This is crucial for median finding and prefix sum calculation.
    X_coords.sort()
    Y_coords.sort()

    # Calculate prefix sums for X and Y coordinates.
    # PX[k] = sum of X_coords[0]...X_coords[k-1]
    PX = [0] * (N + 1)
    for i in range(N):
        PX[i+1] = PX[i] + X_coords[i]

    PY = [0] * (N + 1)
    for i in range(N):
        PY[i+1] = PY[i] + Y_coords[i]

    # Function to calculate S(val) = sum |val - coord_i| for a list of coordinates `coords`.
    # Uses prefix sums for O(log N) calculation time after sorting.
    def calc_S(val, coords, prefix_sums, N):
        # Find k = number of coordinates <= val using binary search (`bisect_right`).
        k = bisect.bisect_right(coords, val)
        # The formula derives from splitting the sum into coords[i] <= val and coords[i] > val.
        # S(val) = sum_{i=1..k} (val - coords[i]) + sum_{i=k+1..N} (coords[i] - val)
        # This simplifies using prefix sums:
        res = (k * val - prefix_sums[k]) + ((prefix_sums[N] - prefix_sums[k]) - (N - k) * val)
        # Further simplified to:
        # res = (2 * k - N) * val + prefix_sums[N] - 2 * prefix_sums[k]
        # This is the form used in the code:
        res = (2 * k - N) * val + prefix_sums[N] - 2 * prefix_sums[k]
        return res

    # Determine reasonable bounds for coordinate search.
    # Based on problem constraints and analysis, max relevant coordinate value is approx +/- (10^6 + D/1).
    # Use a safe bound including a buffer.
    COORD_BOUND = 2 * 10**6 + 10 
    MIN_COORD = -COORD_BOUND
    MAX_COORD = COORD_BOUND

    # Find a median x-coordinate. The median minimizes Sx(x).
    if N % 2 == 1:
        x_m = X_coords[N // 2]
    else:
        # For even N, any integer value in [X[N//2 - 1], X[N//2]] is a median.
        # We pick one point (e.g., X[N//2 - 1]) as a representative median for binary search pivots.
         x_m = X_coords[N // 2 - 1] 
    
    # Find a median y-coordinate and calculate the minimum possible value of Sy(y).
    if N % 2 == 1:
        y_m = Y_coords[N // 2]
    else:
        y_m = Y_coords[N // 2 - 1]
    Sy_min = calc_S(y_m, Y_coords, PY, N)

    # Calculate the minimum possible value of Sx(x).
    Sx_min_check = calc_S(x_m, X_coords, PX, N)
    
    # Check if the minimum possible total distance sum exceeds D.
    # If Sx_min + Sy_min > D, no point (x, y) can satisfy the condition.
    if Sx_min_check + Sy_min > D:
       print(0)
       return

    # Binary search to find the range [x_low, x_high] of x-coordinates potentially satisfying Sx(x) <= D.
    # Any x outside this range cannot be part of a solution since Sx(x) + Sy(y) >= Sx(x) > D (as Sy(y) >= 0).
    
    # Find x_low: the smallest integer x such that Sx(x) <= D
    low = MIN_COORD
    high = x_m
    x_low = x_m # Initialize with x_m, a known point where Sx(x) <= D (since Sx_min + Sy_min <= D implies Sx_min <= D)
    
    while low <= high:
        mid = (low + high) // 2
        if calc_S(mid, X_coords, PX, N) <= D:
            x_low = mid # Found a potential candidate, try smaller x
            high = mid - 1
        else:
            low = mid + 1 # Need larger x 

    # Find x_high: the largest integer x such that Sx(x) <= D
    low = x_m
    high = MAX_COORD
    x_high = x_m # Initialize with x_m

    while low <= high:
        mid = (low + high) // 2
        if calc_S(mid, X_coords, PX, N) <= D:
            x_high = mid # Found a potential candidate, try larger x
            low = mid + 1
        else:
             high = mid - 1 # Need smaller x

    # Precompute counts of distinct X coordinates. This allows O(1) update of 'k' in the loop.
    counts_X = Counter(X_coords)

    ans = 0 # Initialize total count of valid integer points (x, y)
    
    # === Initial calculation for x = x_low ===
    current_Sx = calc_S(x_low, X_coords, PX, N)
    Dy = D - current_Sx # Remaining budget for Sy(y)

    # Initialize the y-range state [y_low_curr, y_high_curr]. 
    # Using (y_m+1, y_m) represents an empty range centered around y_m.
    # This state indicates that currently no valid y range is known.
    y_low_curr = y_m + 1 
    y_high_curr = y_m 

    # If the budget Dy allows for any y (i.e., Dy >= minimum Sy), find the initial range [y_low, y_high] using Binary Search.
    if Dy >= Sy_min:
        # Find initial y_low: smallest y such that Sy(y) <= Dy
        low = MIN_COORD
        high = y_m
        temp_y_low = y_m + 1 # Default if no solution found for y <= y_m
        
        while low <= high:
            mid = (low + high) // 2
            if calc_S(mid, Y_coords, PY, N) <= Dy:
                 temp_y_low = mid # Found candidate, try smaller y
                 high = mid - 1
            else:
                low = mid + 1 # Need larger y
        
        # Find initial y_high: largest y such that Sy(y) <= Dy
        low = y_m
        high = MAX_COORD
        temp_y_high = y_m - 1 # Default if no solution found for y >= y_m

        while low <= high:
            mid = (low + high) // 2
            if calc_S(mid, Y_coords, PY, N) <= Dy:
                 temp_y_high = mid # Found candidate, try larger y
                 low = mid + 1
            else:
                 high = mid - 1 # Need smaller y

        # If a valid range was found, update current state and add count for x_low.
        if temp_y_low <= temp_y_high:
             y_low_curr = temp_y_low
             y_high_curr = temp_y_high
             ans += (y_high_curr - y_low_curr + 1)
        # else: range remains empty, initial ans=0 reflects this.

    # Initialize k = count of X_coords[i] <= x_low. This is needed for the first slope calculation inside the loop.
    k = bisect.bisect_right(X_coords, x_low)
    
    # === Iterate through x values from x_low to x_high - 1 ===
    # In each iteration, we calculate the contribution for x+1 based on the state at x.
    for x in range(x_low, x_high):
        
        # The slope of Sx function at point x determines the change Sx(x+1) - Sx(x).
        slope_x = 2*k - N 
        current_Sx += slope_x # Update Sx: Sx(x+1) = Sx(x) + slope_x * ( (x+1) - x )

        # Update k for the next point (x+1). k increases by the count of points exactly equal to x+1.
        # Use counts_X.get(x+1, 0) for O(1) lookup, returns 0 if x+1 is not in X_coords.
        k += counts_X.get(x+1, 0) 
        
        # Update remaining budget for Sy(y) for the point x+1
        Dy = D - current_Sx 

        # If budget is less than minimum possible Sy, the y range must be empty for x+1.
        if Dy < Sy_min:
             # Set state to reflect empty range. No points to add for this x+1.
             y_low_curr = y_m + 1
             y_high_curr = y_m
             continue # Proceed to the next x value

        # Adaptively update the y range [y_low_curr, y_high_curr] based on the new budget Dy.
        # This avoids full binary search for y range in each step.
        # Start search from the previous range bounds [y_low_curr, y_high_curr].
        
        # Expand/shrink the high end (y_high_curr)
        # Expand upwards while feasible (Sy(y+1) <= Dy) and within MAX_COORD bound.
        while y_high_curr < MAX_COORD and calc_S(y_high_curr + 1, Y_coords, PY, N) <= Dy:
             y_high_curr += 1
        # Shrink downwards if current y_high is now too large (Sy(y) > Dy).
        # The check y_high_curr >= y_low_curr ensures we don't cross boundaries incorrectly during shrink.
        while y_high_curr >= y_low_curr and calc_S(y_high_curr, Y_coords, PY, N) > Dy: 
             y_high_curr -= 1

        # Expand/shrink the low end (y_low_curr)
        # Expand downwards while feasible (Sy(y-1) <= Dy) and within MIN_COORD bound.
        while y_low_curr > MIN_COORD and calc_S(y_low_curr - 1, Y_coords, PY, N) <= Dy:
             y_low_curr -= 1
        # Shrink upwards if current y_low is now too small (Sy(y) > Dy).
        # Check against the possibly updated y_high_curr to maintain validity (y_low <= y_high).
        while y_low_curr <= y_high_curr and calc_S(y_low_curr, Y_coords, PY, N) > Dy: 
            y_low_curr += 1
        
        # If the updated range [y_low_curr, y_high_curr] is valid (low <= high), add its size to total count.
        if y_low_curr <= y_high_curr:
            ans += (y_high_curr - y_low_curr + 1)
        # else: The range for this x+1 is empty. Add 0. The state (y_low_curr > y_high_curr) is kept for the next iteration.

    # Print the final total count
    print(ans)

# Execute the solve function
solve()