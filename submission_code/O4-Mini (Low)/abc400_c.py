import sys
import threading

def main():
    import sys
    from math import isqrt

    N = int(sys.stdin.readline().strip())
    total = 0
    v = 1
    # Loop over exponent v of 2, while N // 2^v >= 1
    while True:
        # Compute floor(N / 2^v)
        m = N >> v
        if m == 0:
            break
        # Maximum t such that t^2 <= m
        t_max = isqrt(m)
        # We only allow odd t >= 1
        # Count of odd t from 1 to t_max is (t_max + 1)//2
        total += (t_max + 1) // 2
        v += 1

    print(total)

if __name__ == "__main__":
    main()