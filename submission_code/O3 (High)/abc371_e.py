import sys

def main() -> None:
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    it = iter(data)
    N = int(next(it))
    A = [int(next(it)) for _ in range(N)]

    # positions[v] keeps all indices (1-based) where value v appears
    positions = [[] for _ in range(N + 1)]
    for idx, val in enumerate(A, 1):          # indices are 1 … N
        positions[val].append(idx)

    total_sub = N * (N + 1) // 2              # total number of sub-arrays
    ans = 0

    for v in range(1, N + 1):
        if not positions[v]:
            continue
        pos = positions[v]

        # add sentinels 0 and N+1 to compute gaps easily
        prev = 0
        gap_sum = 0
        for p in pos:
            gap = p - prev - 1
            gap_sum += gap * (gap + 1) // 2
            prev = p
        # gap after last occurrence
        gap = (N + 1) - prev - 1
        gap_sum += gap * (gap + 1) // 2

        ans += total_sub - gap_sum            # sub-arrays that contain v

    print(ans)

if __name__ == "__main__":
    main()