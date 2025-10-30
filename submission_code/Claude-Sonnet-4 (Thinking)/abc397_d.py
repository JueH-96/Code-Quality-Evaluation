import math

N = int(input())

max_d = int(N ** (1/3)) + 2

found = False
for d in range(1, max_d + 1):
    if d * d * d >= N:
        break
    if N % d == 0:
        quotient = N // d
        discriminant = -3 * d * d + 12 * quotient
        
        if discriminant >= 0:
            sqrt_discriminant_float = math.sqrt(discriminant)
            sqrt_discriminant = int(sqrt_discriminant_float + 0.5)
            
            if sqrt_discriminant * sqrt_discriminant == discriminant:
                numerator = -3 * d + sqrt_discriminant
                if numerator % 6 == 0 and numerator > 0:
                    y = numerator // 6
                    if y > 0:
                        x = y + d
                        if x * x * x - y * y * y == N:
                            print(x, y)
                            found = True
                            break

if not found:
    print(-1)