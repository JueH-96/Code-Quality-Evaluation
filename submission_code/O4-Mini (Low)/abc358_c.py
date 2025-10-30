def main():
    import sys
    input = sys.stdin.readline

    N, M = map(int, input().split())
    stands = [input().strip() for _ in range(N)]
    # Convert each stand's flavors to a bitmask
    masks = []
    for s in stands:
        mask = 0
        for j, c in enumerate(s):
            if c == 'o':
                mask |= (1 << j)
        masks.append(mask)

    full = (1 << M) - 1
    ans = N + 1

    # Try all subsets of stands
    for subset in range(1, 1 << N):
        cnt = subset.bit_count()
        if cnt >= ans:
            continue
        cover = 0
        # Union the flavor-sets of chosen stands
        b = subset
        i = 0
        while b:
            if b & 1:
                cover |= masks[i]
            b >>= 1
            i += 1
            if cover == full:
                break
        if cover == full:
            ans = cnt

    print(ans)

if __name__ == "__main__":
    main()