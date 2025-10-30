def main():
    import sys
    import math
    
    D = int(sys.stdin.readline())
    
    # We want to find min |x^2 + y^2 - D|.
    # Let x range from 0 to floor(sqrt(D)).
    # For each x, find the y that makes x^2 + y^2 as close to D as possible:
    # y ~ sqrt(D - x^2). We check floor(...) and floor(...)+1 for minimal absolute difference.
    # Then also consider x = floor(sqrt(D))+1 with y=0, in case (x^2 - D) is smaller.
    
    xMax = int(math.isqrt(D))
    best_diff = D  # Initial guess (e.g., x=y=0 => difference=D).
    
    for x in range(xMax + 1):
        r = D - x*x
        if r < 0:
            break  # No need to continue, x^2 is already too large
        y = int(math.isqrt(r))
        
        # Check y= floor(sqrt(r)) and y+1
        for candidate in [y, y + 1]:
            val = x*x + candidate*candidate
            diff = abs(val - D)
            if diff < best_diff:
                best_diff = diff
                if best_diff == 0:  # Can't do better than 0
                    print(0)
                    return
    
    # Now check x = xMax+1 with y=0 (in case x^2 just over D is closer)
    x2 = (xMax + 1) * (xMax + 1)
    diff = x2 - D
    if diff < best_diff:
        best_diff = diff
    
    print(best_diff)

# Do not forget to call main()!
main()