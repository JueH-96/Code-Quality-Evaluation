import sys
import threading

def main():
    import sys
    input = sys.stdin.readline

    N = int(input())
    A = [0] + list(map(int, input().split()))
    # pos[x] = current index of value x in A
    pos = [0] * (N + 1)
    for idx in range(1, N + 1):
        pos[A[idx]] = idx

    ops = []
    # For each position i, if A[i] != i, swap A[i] with the position of i
    for i in range(1, N + 1):
        if A[i] != i:
            p = pos[i]
            # record operation
            ops.append((i, p))
            # perform swap in A
            Ai, Ap = A[i], A[p]
            A[i], A[p] = Ap, Ai
            # update positions
            pos[Ai] = p
            pos[Ap] = i

    # Output
    out = [str(len(ops))]
    for i, j in ops:
        out.append(f"{i} {j}")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()