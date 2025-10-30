import sys
import threading

def main():
    import sys
    from math import gcd

    data = sys.stdin
    T = int(data.readline())
    out = []
    for _ in range(T):
        line = data.readline().split()
        if not line:
            # in case of blank line
            line = data.readline().split()
        N = int(line[0]); K = int(line[1])
        # Compute g = gcd(N, 2*K)
        g = gcd(N, 2*K)
        # We can color all points iff g <= 2 and (N//g) is odd
        if g <= 2 and ((N // g) & 1):
            out.append("Yes")
        else:
            out.append("No")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()