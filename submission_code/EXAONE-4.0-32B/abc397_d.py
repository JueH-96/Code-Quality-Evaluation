import math

def main():
    N = int(input().strip())
    d = 1
    found = False
    while d * d * d <= 4 * N:
        d3 = d * d * d
        if d3 > 4 * N:
            break
        discriminant_val = 3 * d * (4 * N - d3)
        if discriminant_val < 0:
            d += 1
            continue
        
        root = math.isqrt(discriminant_val)
        if root * root != discriminant_val:
            d += 1
            continue
            
        numerator = -3 * d * d + root
        if numerator <= 0:
            d += 1
            continue
            
        if numerator % (6 * d) == 0:
            y = numerator // (6 * d)
            x = y + d
            print(f"{x} {y}")
            found = True
            break
            
        d += 1
        
    if not found:
        print(-1)

if __name__ == '__main__':
    main()