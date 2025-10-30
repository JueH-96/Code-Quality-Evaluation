def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    S = [input().strip() for _ in range(N)]

    # Convert each stand's flavors into a bitmask
    masks = []
    for s in S:
        mask = 0
        for j, ch in enumerate(s):
            if ch == 'o':
                mask |= 1 << j
        masks.append(mask)

    full = (1 << M) - 1
    ans = N + 1

    # Try every subset of stands
    for subset in range(1 << N):
        combined = 0
        cnt = 0
        # Build the union of flavors and count stands used
        for i in range(N):
            if (subset >> i) & 1:
                combined |= masks[i]
                cnt += 1
        # If this subset covers all flavors, update answer
        if combined == full and cnt < ans:
            ans = cnt

    print(ans)


if __name__ == "__main__":
    main()