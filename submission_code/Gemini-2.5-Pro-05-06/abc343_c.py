import math

def is_palindrome(s_val):
    """Checks if a string is a palindrome."""
    return s_val == s_val[::-1]

def solve():
    N = int(input())

    # Calculate x_limit = floor(N^(1/3))
    # This is the largest integer x such that x^3 <= N.
    
    # Initial estimate for x_limit. int() truncates towards zero.
    # For positive numbers, int(float_val) is generally floor(float_val).
    # However, N**(1.0/3.0) might be slightly off due to floating point precision.
    # e.g. true N^(1/3) = 3.0, float might be 2.999...96, int() gives 2.
    # or true N^(1/3) = 2.999..., float might be 3.000...01, int() gives 3.
    
    x_val = int(N**(1.0/3.0))

    # Adjust x_val upwards if it's an underestimate.
    # (x_val + 1)^3 <= N implies x_val was too small.
    # Loop continues as long as (x_val+1) is a better candidate for floor(N^(1/3)).
    # Python integers handle arbitrary size, so (x_val+1)**3 is exact.
    while (x_val + 1)**3 <= N:
        x_val += 1
        
    # Adjust x_val downwards if it's an overestimate.
    # x_val^3 > N implies x_val was too large.
    # This step ensures x_val^3 <= N.
    # This might be needed if initial int(N**(1/3.0)) rounded up or the previous loop overshot
    # (unlikely with standard float precision and int truncation, but safe to have).
    while x_val**3 > N:
        # x_val must be positive as N is positive. If x_val becomes 0 here,
        # then 0^3 > N, which means N would have to be negative (not allowed).
        # So x_val will remain >= 1 if N >= 1.
        x_val -= 1

    x_limit = x_val
    
    # Iterate x downwards from x_limit to 1
    # Smallest N is 1, so x_limit will be at least 1.
    # The loop range(x_limit, 0, -1) correctly iterates from x_limit down to 1.
    for x in range(x_limit, 0, -1):
        cube = x**3
        
        # K=cube. Condition K <= N is implicitly satisfied because x <= x_limit and x_limit^3 <= N.
        
        if is_palindrome(str(cube)):
            print(cube)
            return # Found the largest palindromic cube, so exit.

solve()