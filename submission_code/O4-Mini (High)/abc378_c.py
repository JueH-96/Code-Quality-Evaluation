import sys
import threading

def main():
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = [int(next(it)) for _ in range(n)]
    last_pos = {}
    B = [0] * n
    for i, a in enumerate(A, start=1):
        if a in last_pos:
            B[i-1] = last_pos[a]
        else:
            B[i-1] = -1
        last_pos[a] = i
    # output
    out = ' '.join(str(x) for x in B)
    sys.stdout.write(out)

if __name__ == "__main__":
    main()