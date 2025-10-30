import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    n = int(data[0])
    A = [0] * (n + 1)
    pos = [0] * (n + 1)
    idx = 1
    for i in range(1, n + 1):
        v = int(data[idx]); idx += 1
        A[i] = v
        pos[v] = i

    swaps = []
    # For each position i, if A[i] != i, swap with the position of i
    for i in range(1, n + 1):
        if A[i] != i:
            j = pos[i]
            # record the swap
            swaps.append((i, j))
            # perform swap in A
            A[i], A[j] = A[j], A[i]
            # update positions in pos
            pos[A[j]] = j
            pos[A[i]] = i

    # Output
    out = [str(len(swaps))]
    for i, j in swaps:
        out.append(f"{i} {j}")
    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()