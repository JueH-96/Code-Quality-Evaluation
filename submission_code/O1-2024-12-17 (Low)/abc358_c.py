def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    S = data[2:]

    # Each stand's flavors can be represented as a bitmask
    masks = []
    for i in range(N):
        bit = 0
        for j, ch in enumerate(S[i]):
            if ch == 'o':
                bit |= (1 << j)
        masks.append(bit)

    # We want the full mask of all M flavors
    full_mask = (1 << M) - 1

    # The answer is at most N, so we can try subsets
    # in increasing size to find the minimal size.
    from itertools import combinations

    # For each size from 1 to N
    for size in range(1, N+1):
        for combo in combinations(range(N), size):
            combined = 0
            for idx in combo:
                combined |= masks[idx]
            if combined == full_mask:
                print(size)
                return

# Do not forget to call main()
if __name__ == "__main__":
    main()