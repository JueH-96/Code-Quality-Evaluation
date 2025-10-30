import math

def main():
    N = int(input().strip())
    d = 1
    while d * d * d <= N:
        if N % d != 0:
            d += 1
            continue
        
        M = N // d
        disc = 12 * M - 3 * d * d
        root = math.isqrt(disc)
        if root * root == disc:
            if root >= 3 * d:
                if (root - 3 * d) % 6 == 0:
                    y = (root - 3 * d) // 6
                    if y > 0:
                        print(f"{d + y} {y}")
                        return
        d += 1
    
    print(-1)

if __name__ == '__main__':
    main()