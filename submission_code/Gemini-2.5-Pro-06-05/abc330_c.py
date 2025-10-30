import math

def solve():
    """
    Reads a positive integer D from standard input, calculates the minimum
    value of |x^2 + y^2 - D| for non-negative integers x and y,
    and prints the result to standard output.
    """
    try:
        D = int(input())
    except (IOError, ValueError):
        # In a controlled environment like a programming contest,
        # input is guaranteed to be valid. This is for robustness.
        return

    # Initialize `ans` with a safe upper bound. For x=0, y=0, the difference is D.
    ans = D

    # We iterate through x and for each x, find the best corresponding y.
    x = 0
    while True:
        x_sq = x * x

        # Optimization: If x^2 alone is further from D than our current best answer,
        # and x^2 > D, we can stop. `x^2 - D > ans` implies `x^2 > D + ans`.
        if x_sq > D + ans:
            break

        rem = D - x_sq
        
        if rem >= 0:
            # We need to find a perfect square y^2 closest to rem.
            # The best integer y will be near sqrt(rem).
            # We check the integers on both sides of sqrt(rem).
            y_approx = int(math.sqrt(rem))
            
            # Candidate 1: y = floor(sqrt(rem))
            # The difference is |x^2 + y_approx^2 - D| = |rem - y_approx^2|
            ans = min(ans, rem - y_approx * y_approx)
            
            # Candidate 2: y = floor(sqrt(rem)) + 1
            y_next = y_approx + 1
            # The difference is |x^2 + y_next^2 - D| = |y_next^2 - rem|
            ans = min(ans, y_next * y_next - rem)
        else:  # rem < 0, which means x_sq > D
            # The closest non-negative y^2 to a negative number is 0 (for y=0).
            # The difference is |x^2 + 0 - D| = x_sq - D = -rem.
            ans = min(ans, -rem)

        # If we find a perfect match, ans becomes 0, and we can't do better.
        if ans == 0:
            break
            
        x += 1
    
    print(ans)

solve()