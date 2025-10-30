import sys
import threading

def main():
    import sys
    from math import gcd
    data = sys.stdin.read().split()
    it = iter(data)
    T = int(next(it))
    out = []
    for _ in range(T):
        n = int(next(it)); k = int(next(it))
        # Compute g = gcd(n, 2*k).
        # If g == 1 (only possible when n odd) or g == 2 (when n even and gcd exactly 2)
        # then all points can be colored; otherwise not.
        g = gcd(n, 2*k)
        if g <= 2:
            out.append("Yes")
        else:
            out.append("No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()