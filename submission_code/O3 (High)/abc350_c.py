import sys

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N = data[0]
    A = data[1:1 + N]

    # pos[v] = current index (0-based) of value v in A
    pos = [0] * (N + 1)
    for idx, val in enumerate(A):
        pos[val] = idx

    swaps = []                       # list of performed swaps (1-based indices)

    for i in range(N - 1):           # we never need to touch the last position
        correct_val = i + 1
        if A[i] == correct_val:      # already in the right place
            continue

        j = pos[correct_val]         # position where the needed value currently is (j > i)

        # record the swap (convert to 1-based indices)
        swaps.append((i + 1, j + 1))

        # perform the swap in A
        val_i = A[i]
        val_j = A[j]                 # this is 'correct_val'
        A[i], A[j] = val_j, val_i

        # update positions
        pos[val_j] = i
        pos[val_i] = j

    out_lines = [str(len(swaps))]
    out_lines.extend(f"{i} {j}" for i, j in swaps)
    sys.stdout.write("
".join(out_lines))

if __name__ == "__main__":
    main()