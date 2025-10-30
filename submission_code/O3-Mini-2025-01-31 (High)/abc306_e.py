def main():
    import sys
    data = sys.stdin.buffer.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    Q = int(next(it))
    
    # Read updates and record all candidate numbers. 
    # (We must include the initial value 0 as well.)
    updates = [None] * Q
    vals = {0}
    for j in range(Q):
        x = int(next(it))
        y = int(next(it))
        updates[j] = (x - 1, y)  # convert to 0-indexed
        vals.add(y)
    
    # Coordinate compress the possible values.
    # We sort in descending order so that rank 1 holds the highest value.
    comp = sorted(vals, reverse=True)
    M = len(comp)
    mp = {v: i + 1 for i, v in enumerate(comp)}   # mapping value -> rank (1-indexed)
    
    # We will use 2 Fenwick Trees (BITs):
    # BIT_count will record the frequency of each rank.
    # BIT_sum will record the total sum contributed by that rank.
    BIT_count = [0] * (M + 1)
    BIT_sum = [0] * (M + 1)
    
    # Standard BIT update and query functions.
    def bit_update(BIT, i, delta):
        while i <= M:
            BIT[i] += delta
            i += i & -i
            
    def bit_query(BIT, i):
        s = 0
        while i:
            s += BIT[i]
            i -= i & -i
        return s

    # BIT_find finds the smallest index pos such that the cumulative frequency
    # (i.e. prefix sum of BIT_count) is at least k.
    bit_max = 1 << (M.bit_length() - 1)
    def bit_find(k):
        pos = 0
        bit = bit_max
        while bit:
            nxt = pos + bit
            if nxt <= M and BIT_count[nxt] < k:
                k -= BIT_count[nxt]
                pos = nxt
            bit //= 2
        return pos + 1

    # Initially, all N entries are 0.
    init_rank = mp[0]
    for _ in range(N):
        bit_update(BIT_count, init_rank, 1)
        # For BIT_sum, adding 0 makes no change.
        
    # current will store the value at each index.
    current = [0] * N
    out_lines = []
    
    # Process each update.
    for (i, new_val) in updates:
        old_val = current[i]
        if old_val != new_val:
            # Remove the old value from BITs.
            old_r = mp[old_val]
            bit_update(BIT_count, old_r, -1)
            bit_update(BIT_sum, old_r, -old_val)
            # Add the new value in BITs.
            new_r = mp[new_val]
            bit_update(BIT_count, new_r, 1)
            bit_update(BIT_sum, new_r, new_val)
            current[i] = new_val
        # Now compute f(A): the sum of the K largest values.
        #
        # Our BIT_count is arranged by rank (with 1 = highest value).
        # We perform a “find kth” query: let pos be the smallest rank index 
        # such that the cumulative frequency is >= K.
        pos = bit_find(K)
        # Get the count and sum for all ranks strictly better (i.e. ranks 1..pos-1)
        cnt = bit_query(BIT_count, pos - 1)
        ssum = bit_query(BIT_sum, pos - 1)
        # For rank pos we only need part of its frequency.
        need = K - cnt
        # comp is 0-indexed; comp[pos-1] gives the actual value for rank "pos".
        ans = ssum + need * comp[pos - 1]
        out_lines.append(str(ans))
    
    sys.stdout.write("
".join(out_lines))

if __name__ == '__main__':
    main()