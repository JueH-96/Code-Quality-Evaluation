import sys
import threading

def main():
    import sys

    data = sys.stdin.read().strip().split()
    it = iter(data)
    n = int(next(it))
    P = [int(next(it)) for _ in range(n)]
    # Build a mapping from person number to their position in line (0-based index)
    pos = {}
    for idx, person in enumerate(P):
        pos[person] = idx

    q = int(next(it))
    out = []
    for _ in range(q):
        a = int(next(it))
        b = int(next(it))
        # Compare positions: smaller index means further to the front
        if pos[a] < pos[b]:
            out.append(str(a))
        else:
            out.append(str(b))

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()