import sys
import threading

def main():
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]
    M = int(next(it))
    B = [int(next(it)) for _ in range(M)]
    L = int(next(it))
    C = [int(next(it)) for _ in range(L)]
    Q = int(next(it))
    X = [int(next(it)) for _ in range(Q)]

    # Precompute all possible sums of A and B
    AB = set()
    for a in A:
        for b in B:
            AB.add(a + b)

    # Precompute all possible sums of (A + B) + C
    ABC = set()
    for s in AB:
        for c in C:
            ABC.add(s + c)

    out = []
    for x in X:
        if x in ABC:
            out.append("Yes")
        else:
            out.append("No")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()