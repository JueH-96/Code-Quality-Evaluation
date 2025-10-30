def isqrt(n):
    if n < 2:
        return n
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x

N = int(input())

# Find max_d such that d^3 < N
max_d = 1
while max_d <= 10**6 and (max_d + 1) ** 3 < N:
    max_d += 1

for d in range(1, max_d + 1):
    if N % d == 0:
        q = N // d
        
        discriminant = 12 * q - 3 * d * d
        
        if discriminant >= 0:
            sqrt_disc = isqrt(discriminant)
            if sqrt_disc * sqrt_disc == discriminant:
                numerator = -3 * d + sqrt_disc
                if numerator > 0 and numerator % 6 == 0:
                    y = numerator // 6
                    x = y + d
                    print(x, y)
                    exit()

print(-1)