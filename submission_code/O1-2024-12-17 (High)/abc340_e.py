def main():
    import sys
    input = sys.stdin.readline
    
    N, M = map(int, input().split())
    A = list(map(int, input().split()))
    B = list(map(int, input().split()))
    
    # Fenwick (Binary Indexed) Tree for range-update + point-query
    fenwicks = [0]*(N+1)  # 1-based indexing internally
    
    # Add val to fenwicks[i], i in [1..N]
    def fenwicks_add(i, val):
        while i <= N:
            fenwicks[i] += val
            i += i & -i
    
    # Returns sum of fenwicks up to i (inclusive), i in [1..N]
    def fenwicks_sum(i):
        s = 0
        while i > 0:
            s += fenwicks[i]
            i -= i & -i
        return s
    
    # Point query on index x (0-based), returns how much has been added to x
    def point_query(x):
        return fenwicks_sum(x+1)
    
    # Range update [l, r] (0-based, inclusive) of +1
    def range_update(l, r):
        # Add +1 starting at l
        fenwicks_add(l+1, 1)
        # Subtract 1 after r
        if r+1 < N:
            fenwicks_add(r+2, -1)
    
    # Handle wrap-around if start > end
    def partial_range_update(start, end):
        if start <= end:
            range_update(start, end)
        else:
            # Wraps around
            range_update(start, N-1)
            range_update(0, end)
    
    # Track how many balls in total have been removed from each box
    removals = [0]*N
    
    # Track how many full-cycles worth of increments have been applied to every box
    globalAdd = 0
    
    # Process M operations in order
    for b in B:
        # Current partial increments on box b
        partialVal = point_query(b)
        # How many balls are in box b right now
        currentBalls = A[b] + globalAdd + partialVal - removals[b]
        
        # Remove them all
        K = currentBalls
        removals[b] += K
        
        # Distribute K balls among consecutive boxes
        # Each box gets K//N, plus 1 for the first (K mod N) boxes
        q, r = divmod(K, N)
        globalAdd += q
        if r > 0:
            start = (b + 1) % N
            end = (b + r) % N
            partial_range_update(start, end)
    
    # Compute final result
    ans = []
    for i in range(N):
        pval = point_query(i)
        # Final number of balls = initial + full-cycles + partial-increments - removed
        final_val = A[i] + globalAdd + pval - removals[i]
        ans.append(str(final_val))
    
    print(" ".join(ans))

# Call main() at the end
if __name__ == "__main__":
    main()