import sys
def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    try:
        N = int(next(it))
    except StopIteration:
        return
    # Read P array; 1-based indexing
    P = [0] * (N + 1)
    for i in range(1, N + 1):
        P[i] = int(next(it))
    # Fenwick tree (BIT) storing counts of free slots; initialize all ones
    tree = [0] * (N + 1)
    for i in range(1, N + 1):
        tree[i] = i & -i
    # Answer array: final positions -> value
    ans = [0] * (N + 1)
    # Precompute highest power-of-two >= N for BIT binary search
    bitMaskMax = 1 << (N.bit_length())
    # Process insertions in reverse: for i = N down to 1
    for i in range(N, 0, -1):
        k = P[i]
        pos = 0
        bitMask = bitMaskMax
        # Find the k-th free position
        while bitMask:
            nxt = pos + bitMask
            if nxt <= N and tree[nxt] < k:
                k -= tree[nxt]
                pos = nxt
            bitMask >>= 1
        finalPos = pos + 1
        ans[finalPos] = i
        # Mark that slot as occupied: subtract 1 at finalPos in BIT
        j = finalPos
        while j <= N:
            tree[j] -= 1
            j += j & -j
    # Output the final array
    out = sys.stdout
    write = out.write
    write(' '.join(str(ans[i]) for i in range(1, N + 1)))

if __name__ == '__main__':
    main()