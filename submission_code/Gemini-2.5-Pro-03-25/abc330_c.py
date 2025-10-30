# YOUR CODE HERE
import math
import sys

def solve():
    # Read the positive integer D from standard input
    D = int(sys.stdin.readline())
    
    # Initialize the minimum difference found so far. 
    # The value D itself is a safe upper bound, achieved by x=0, y=0: |0^2+0^2-D| = D.
    min_diff = D 
    
    # Calculate the integer square root of D using math.isqrt for precision with large integers.
    # This value, floor(sqrt(D)), serves as a limit for our main search range for x.
    x_limit = math.isqrt(D)
    
    # We need to check values of x starting from 0. The necessary range is up to floor(sqrt(D)) + 1.
    # The loop range(x_limit + 2) iterates through x = 0, 1, ..., x_limit, x_limit + 1.
    for x in range(x_limit + 2): 
        # Calculate x^2 efficiently. Python handles large integers automatically.
        x_squared = x*x
        
        # Calculate T = D - x^2. This represents the target value for y^2.
        T = D - x_squared
        
        # Check if x^2 > D. This happens when T < 0.
        if T < 0:
            # Case 1: x^2 > D.
            # This condition is first met when x = floor(sqrt(D)) + 1.
            # For a fixed x such that x^2 > D, the expression |x^2+y^2-D| simplifies to x^2+y^2-D,
            # because x^2+y^2 > D for any non-negative y.
            # This expression is minimized when y=0. The minimum value is x^2 - D.
            diff = x_squared - D  # This is also equal to -T
            min_diff = min(min_diff, diff)
            
            # For any subsequent x' > x, the minimum difference for x' (which is (x')^2 - D)
            # will be strictly greater than the minimum difference for x (which is x^2 - D).
            # Therefore, we don't need to check larger values of x. We can stop the search.
            break 
            
        # Case 2: x^2 <= D. This means T >= 0.
        # We need to find a non-negative integer y such that y^2 is closest to T.
        # The candidates for y^2 are squares of floor(sqrt(T)) and ceil(sqrt(T)).
        # Let y = floor(sqrt(T)). The candidates are y^2 and (y+1)^2.
        y = math.isqrt(T)
        
        # Candidate sum 1: S1 = x^2 + y^2.
        # Since y = floor(sqrt(T)), we have y^2 <= T = D - x^2.
        # Rearranging gives x^2 + y^2 <= D.
        # The absolute difference |S1 - D| is D - S1.
        # D - S1 = D - (x^2 + y^2) = (D - x^2) - y^2 = T - y^2.
        y_squared = y*y
        diff1 = T - y_squared 
        min_diff = min(min_diff, diff1)
        
        # Candidate sum 2: S2 = x^2 + (y+1)^2.
        # Since y = floor(sqrt(T)), we have y+1 > sqrt(T).
        # Squaring gives (y+1)^2 > T = D - x^2.
        # Rearranging gives x^2 + (y+1)^2 > D.
        # The absolute difference |S2 - D| is S2 - D.
        # S2 - D = (x^2 + (y+1)^2) - D = (y+1)^2 - (D - x^2) = (y+1)^2 - T.
        y_plus_1 = y + 1
        y_plus_1_squared = y_plus_1 * y_plus_1
        diff2 = y_plus_1_squared - T 
        min_diff = min(min_diff, diff2)
        
    # After checking all necessary values of x, print the overall minimum difference found.
    print(min_diff)

# Execute the solve function to run the program
solve()