def main():
    import sys
    data = sys.stdin.buffer.read().split()
    if not data:
        return
    n = int(data[0])
    # P is given in insertion order; P[i-1] is for the insertion of i.
    P = [int(x) for x in data[1:]]
    
    # We will create a BIT (Fenwick Tree) for positions 1..n.
    # Initially, all positions are free (value = 1).
    BIT = [0] * (n+1)
    for i in range(1, n+1):
        BIT[i] = 1
    # Build BIT in O(n) time.
    for i in range(1, n+1):
        j = i + (i & -i)
        if j <= n:
            BIT[j] += BIT[i]
    
    # BIT update: add delta to position i.
    def fenw_update(i, delta):
        while i <= n:
            BIT[i] += delta
            i += i & -i

    # BIT find: find the smallest index i such that the prefix sum BIT[1..i] >= k.
    def fenw_find(k):
        idx = 0
        # Start with the largest power of 2 <= n.
        bit_mask = 1 << (n.bit_length() - 1)
        while bit_mask:
            next_idx = idx + bit_mask
            if next_idx <= n and BIT[next_idx] < k:
                k -= BIT[next_idx]
                idx = next_idx
            bit_mask //= 2
        return idx + 1

    # Prepare the result array.
    res = [0] * n
    # Process numbers from n down to 1.
    # For each i, we use P[i-1] to decide where (among the free slots) to put i.
    for i in range(n, 0, -1):
        pos = fenw_find(P[i-1])
        res[pos-1] = i  # pos is 1-indexed; convert to 0-index
        fenw_update(pos, -1)  # Mark that slot as no longer free.
    
    sys.stdout.write(" ".join(map(str, res)))
    
if __name__ == '__main__':
    main()