def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, M = map(int, data[:2])
    X = list(map(int, data[2:2+M]))
    A = list(map(int, data[2+M:2+2*M]))
    
    # 1) Check total number of stones == N
    total_stones = sum(A)
    if total_stones != N:
        print(-1)
        return
    
    # Pair up and sort by cell index
    pairs = sorted(zip(X, A), key=lambda x: x[0])
    
    # We'll keep track of the running partial sum of stones up to current cell,
    # and compute the cost of shifting in segments.
    
    # partial_stones = sum of stones up to (and including) 'current_cell'
    # last_cell = the last cell we have "processed".
    partial_stones = 0
    last_cell = 0
    cost = 0
    
    # A helper to compute sum of consecutive integers from l to r (inclusive).
    # sum of i for i in [l..r]
    def sum_range(l, r):
        return (r*(r+1)//2) - ((l-1)*l//2)
    
    for (cell, stones) in pairs:
        # From last_cell+1 to cell-1, partial_stones stays the same
        # For i in [last_cell+1..cell-1], D_i = partial_stones - i
        # We must check if min(D_i) in that range is >= 0 (for feasibility).
        # D_i is linear in i with slope -1, so the minimum is at i = cell-1.
        # So we check partial_stones - (cell-1) >= 0  -> partial_stones >= cell-1
        # but only if cell-1 >= last_cell+1 (meaning there's actually a gap).
        if cell - 1 >= last_cell + 1:
            # Check feasibility
            if partial_stones < (cell - 1):
                print(-1)
                return
            # Add cost for that segment
            l = last_cell + 1
            r = cell - 1
            length = (r - l + 1)
            # sum_{i=l..r} [partial_stones - i]
            # = length * partial_stones - sum_{i=l..r} i
            segment_sum = length * partial_stones - sum_range(l, r)
            cost += segment_sum
        
        # Now we are at cell 'cell' itself:
        # Check feasibility D_cell = partial_stones + stones - cell (only if cell < N)
        # But we only need to check D_cell >= 0 if cell < N.  If cell == N, that one is
        # handled at the very end. We'll do a partial check afterwards anyway.
        
        # First, add the new stones in cell 'cell'
        partial_stones += stones
        
        if cell < N:
            if partial_stones < cell:
                print(-1)
                return
            # Cost up to this cell is not added yet; cost for cell itself merges with next segment.
            # We do nothing regarding cost for exactly i=cell, because we handle segments (l+1..r).
        
        last_cell = cell
    
    # Finally, we may have a segment from last_cell+1 to N (if last_cell < N).
    # For i in [last_cell+1..N-1], D_i = partial_stones - i must be >= 0.
    # The minimum again is at i = N-1 in that range, so check partial_stones >= N-1 if last_cell < N-1.
    if last_cell < N:
        if N - 1 >= last_cell + 1:
            if partial_stones < (N - 1):
                print(-1)
                return
            l = last_cell + 1
            r = N - 1
            if r >= l:
                length = (r - l + 1)
                segment_sum = length * partial_stones - sum_range(l, r)
                cost += segment_sum
    
    # Finally, we need partial_stones == N at cell N
    if partial_stones != N:
        print(-1)
        return
    
    # If we reach here, it is feasible. Print the cost.
    print(cost)

# Do not forget to call main()
main()