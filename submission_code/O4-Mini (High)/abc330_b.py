import sys

def main():
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    L = int(next(it))
    R = int(next(it))
    out = []
    for _ in range(n):
        a = int(next(it))
        # clamp a to the interval [L, R]
        if a < L:
            out.append(str(L))
        elif a > R:
            out.append(str(R))
        else:
            out.append(str(a))
    sys.stdout.write(" ".join(out))

main()