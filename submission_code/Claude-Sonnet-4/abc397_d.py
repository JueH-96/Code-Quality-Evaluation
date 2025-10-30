def integer_cube_root(n):
    """Returns the integer cube root of n if it exists, otherwise None"""
    if n == 0:
        return 0
    if n < 0:
        return None
    
    # Initial guess using floating point
    x = int(round(n ** (1/3)))
    
    # Check nearby values due to floating point precision issues
    for candidate in [x-1, x, x+1]:
        if candidate >= 0 and candidate ** 3 == n:
            return candidate
    
    return None

def solve(N):
    # Try different values of y
    max_y = min(1000000, int(N ** (1/3)) + 1000)
    
    for y in range(1, max_y + 1):
        x_cubed = y**3 + N
        x = integer_cube_root(x_cubed)
        
        if x is not None and x > y:
            return x, y
    
    return None

N = int(input())
result = solve(N)

if result is None:
    print(-1)
else:
    x, y = result
    print(x, y)