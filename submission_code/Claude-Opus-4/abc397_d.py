def solve():
    N = int(input())
    
    # Try different values of d (difference between x and y)
    # We only need to check up to a reasonable limit
    # If d is too large, y would be negative
    limit = min(10**6, int(N**(1/3)) + 1000)
    
    for d in range(1, limit + 1):
        if N % d != 0:
            continue
            
        # Check if we can find integer y
        # From d(3y² + 3yd + d²) = N
        # We get 3y² + 3yd + d² - N/d = 0
        
        discriminant = 9 * d * d - 12 * (d * d - N // d)
        if discriminant < 0:
            continue
            
        sqrt_disc = int(discriminant ** 0.5)
        if sqrt_disc * sqrt_disc != discriminant:
            continue
            
        # Check both solutions of quadratic
        if (-3 * d + sqrt_disc) % 6 == 0:
            y = (-3 * d + sqrt_disc) // 6
            if y > 0:
                x = y + d
                if x * x * x - y * y * y == N:
                    print(x, y)
                    return
                    
        if (-3 * d - sqrt_disc) % 6 == 0:
            y = (-3 * d - sqrt_disc) // 6
            if y > 0:
                x = y + d
                if x * x * x - y * y * y == N:
                    print(x, y)
                    return
    
    print(-1)

solve()