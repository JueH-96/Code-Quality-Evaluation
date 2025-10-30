def main():
    import sys
    input = sys.stdin.readline
    
    N = int(input().strip())
    departments = list(map(int, input().split()))
    
    total = sum(departments)
    best = total  # best answer initial upper bound: all in one group
    
    # Loop over all subsets: total 2^N possibilities (N <= 20 so fine)
    # Use bitmask integer from 0 to 2**N - 1 to represent selection of group A.
    # Group B has the remaining departments.
    
    # We can do a simple for loop over bit masks:
    for mask in range(1 << N):
        groupA = 0
        # bit-level iteration
        # Also, we can use a pointer iteration over bits (0 to N-1)
        m = mask
        i = 0
        while m:
            if m & 1:
                groupA += departments[i]
            m >>= 1
            i += 1
        
        groupB = total - groupA
        # we want to minimize the maximum of these sums.
        curr_max = max(groupA, groupB)
        if curr_max < best:
            best = curr_max
            
        # Early exit optimization: if groupA equals half the total then best is best possible.
        # But since total may be odd, we don't know. So no strict early termination.
    sys.stdout.write(str(best))
    
if __name__ == '__main__':
    main()