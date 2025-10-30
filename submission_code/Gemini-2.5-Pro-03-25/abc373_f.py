# YOUR CODE HERE
import sys
from collections import deque

# Function to compute cross product check for CHT upper hull convexity
# Checks if slope(p2, p3) <= slope(p1, p2) where p1, p2, p3 are points ordered by x-coordinate.
# Points are tuples (x, y). x = q', y = Y(q')
# Returns True if the convexity property holds (p2 should potentially stay on the hull), False otherwise (p2 must be removed).
def check_convexity(p1, p2, p3):
    """
    Checks the upper hull convexity condition for three points p1, p2, p3 ordered by x-coordinates.
    Uses cross product to avoid floating point arithmetic and division by zero issues.
    The condition for p2 to potentially stay on the upper hull is slope(p2, p3) <= slope(p1, p2).
    This translates to (y3 - y2) * (x2 - x1) <= (y2 - y1) * (x3 - x2).
    """
    # p = (x, y)
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    # The check below assumes x1 < x2 < x3, which is guaranteed by the algorithm's structure (q' increments).
    # Calculate the cross product components
    val1 = (y3 - y2) * (x2 - x1)
    val2 = (y2 - y1) * (x3 - x2)
    
    # Compare using Python's arbitrary precision integers
    return val1 <= val2

# Function to check optimality for query slope m
# Determines if point p2 provides a better or equal value than p1 for maximizing Y + mX.
# Check: Y2 + m*X2 >= Y1 + m*X1
# Returns True if p1 is dominated or equalled by p2 for this slope m, meaning p1 can be removed from the front of the deque.
def check_query_optimality(p1, p2, m):
    """
    Checks if point p2 yields a higher or equal value than p1 for the line Y = -mX + C, i.e., maximizing C = Y + mX.
    Condition is Y2 + m*X2 >= Y1 + m*X1.
    """
    # p = (x, y)
    x1, y1 = p1
    x2, y2 = p2
    
    # Calculate values using Python's arbitrary precision integers
    val1 = y1 + m * x1
    val2 = y2 + m * x2
    
    return val2 >= val1


def solve():
    N, W = map(int, sys.stdin.readline().split())
    items = []
    for i in range(N):
        items.append(list(map(int, sys.stdin.readline().split())) ) 

    # Use a sufficiently small integer value to represent negative infinity.
    # This value should be lower than any possible achievable happiness.
    # Max happiness can be around 3000 * 10^9 = 3e12. Min happiness can be negative.
    # A value like -10^18 should be safe.
    NEG_INF = -10**18 
    
    # dp[w] stores the maximum happiness achievable with total weight exactly w.
    dp = [NEG_INF] * (W + 1)
    dp[0] = 0 # Base case: 0 weight gives 0 happiness

    # Iterate through each item type
    for item_idx in range(N):
        w_i, v_i = items[item_idx]
        
        # Optimization: If item weight exceeds capacity, it cannot be used.
        if w_i > W: continue
        # Optimization: If item weight is 0, it complicates DP. Problem states w_i >= 1.

        # dp_next will store the DP state after considering item i
        # Initialize with the current dp state. This correctly handles the k_i=0 case (not using item i).
        dp_next = list(dp) 

        # Process states based on remainder modulo w_i
        for r in range(w_i):
            
            # Deque stores indices q' of points currently on the upper convex hull for this remainder class
            hull_indices = deque() 
            # Dictionary stores point coordinates mapping q' -> (q', Y(q'))
            # Y(q') = dp[q' * w_i + r] - q' * v_i - q' * q'
            points = {} 

            # Iterate through possible quotient values q for the current remainder r
            # q corresponds to the number of items 'blocks' of weight w_i
            max_q = (W - r) // w_i
            
            for q in range(max_q + 1):
                current_w = q * w_i + r # Current weight being considered
                
                # Query slope for CHT is 2*q
                query_slope = 2 * q
                
                # --- Convex Hull Trick Query Phase ---
                # Remove points from the front of the deque that are no longer optimal
                # for the current query slope (which is increasing with q).
                while len(hull_indices) >= 2:
                    q1_idx = hull_indices[0]
                    q2_idx = hull_indices[1]
                    
                    # Ensure points exist (defensive check)
                    if q1_idx not in points or q2_idx not in points:
                       # This state suggests an inconsistency. Minimal recovery: remove problematic indices.
                       if q1_idx not in points: hull_indices.popleft()
                       # The loop re-evaluates. If q2_idx also not in points, it might be handled next.
                       # Or perhaps state implies deque has stale indices. Prudence may require clearing/rebuilding.
                       # Assuming points exist based on logic.
                       break # break for safety or proceed with caution

                    p1 = points[q1_idx]
                    p2 = points[q2_idx]
                    
                    # If p2 gives a better or equal result than p1 for the current slope m=2q, then p1 is dominated and can be removed.
                    if check_query_optimality(p1, p2, query_slope):
                       hull_indices.popleft()
                    else:
                        # p1 is still the optimal point among the leftmost hull vertices for this slope.
                        break

                # --- Calculate Happiness Update ---
                # If the hull is not empty, use the optimal point found (front of deque) to potentially update dp_next[current_w]
                if len(hull_indices) > 0:
                    best_q_prime_idx = hull_indices[0]
                     
                    if best_q_prime_idx in points: # Check point exists
                         best_p = points[best_q_prime_idx] # This is the point (q', Y(q')) that maximizes Y + mX
                         
                         # The maximum value from CHT query is Y(best_q') + m * X(best_q')
                         max_val_cht = best_p[1] + query_slope * best_p[0]
                         
                         # Calculate the total happiness for current_w using k = q - best_q' items of type i
                         # Formula: f(q) = max_val_cht + q*v_i - q*q
                         current_happiness = max_val_cht + q * v_i - q * q
                         
                         # Update dp_next[current_w] if this path gives higher happiness
                         dp_next[current_w] = max(dp_next[current_w], current_happiness)


                # --- Convex Hull Trick Add Point Phase ---
                # Check if the state dp[current_w] was reachable using items 1 to i-1.
                if dp[current_w] > NEG_INF: 
                    # Calculate the coordinates for the point corresponding to the current state (q acts as q')
                    # Point coordinates: (X, Y) = (q, dp[current_w] - q*v_i - q*q)
                    Y_q = dp[current_w] - q * v_i - q * q
                    current_point_coords = (q, Y_q)
                    points[q] = current_point_coords # Store the point coordinates

                    # Add the new point to the hull, maintaining the upper hull convexity property.
                    # Remove points from the back of the deque if they violate convexity with the new point.
                    while len(hull_indices) >= 2:
                        q1_idx = hull_indices[-2] # Second to last point index
                        q2_idx = hull_indices[-1] # Last point index
                         
                        # Ensure points exist
                        if q1_idx not in points or q2_idx not in points:
                           # Handle inconsistent state. Could remove bad index.
                           if q2_idx not in points: hull_indices.pop() 
                           # If q1_idx is also bad, implies deeper issue.
                           break # Stop processing this add phase for safety.

                        p1 = points[q1_idx]
                        p2 = points[q2_idx]
                        p3 = current_point_coords # The new point being added
                        
                        # Check if adding p3 maintains convexity (slope p2p3 <= slope p1p2)
                        if not check_convexity(p1, p2, p3): 
                            # Convexity violated: p2 is below the line segment p1p3. Remove p2.
                            hull_indices.pop() 
                        else:
                            # Convexity holds: p2 should remain. p3 will be added after this loop.
                            break 
                    hull_indices.append(q) # Add the index q of the new point to the deque

        # After processing all remainder classes for item i, update the main DP table
        dp = dp_next 

    # --- Final Answer Calculation ---
    # The final answer is the maximum happiness found across all possible weights up to W.
    final_ans = 0
    for w in range(W + 1):
       # Consider only reachable states (happiness > NEG_INF)
       if dp[w] > NEG_INF: 
          final_ans = max(final_ans, dp[w])
    
    # The problem asks for maximum happiness. Since choosing 0 items gives 0 happiness, the answer must be at least 0.
    print(max(0, final_ans))

solve()