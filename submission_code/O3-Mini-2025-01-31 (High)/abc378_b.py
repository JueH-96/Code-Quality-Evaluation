def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    
    # Read the number of garbage types
    N = int(next(it))
    # Read the parameters for each type (indexing from 1 for convenience)
    types = [None] * (N + 1)
    for i in range(1, N + 1):
        q = int(next(it))
        r = int(next(it))
        types[i] = (q, r)
    
    # Read the number of queries
    Q = int(next(it))
    ans = []
    
    for _ in range(Q):
        t = int(next(it))
        d = int(next(it))
        q, r = types[t]
        # Compute the remainder when d is divided by q
        rem = d % q
        
        # If the day d is already a collection day, then the answer is d itself.
        if rem == r:
            collection_day = d
        # If the current remainder is less than r, add the difference.
        elif rem < r:
            collection_day = d + (r - rem)
        # Otherwise, we need to go to the next cycle after d.
        else:
            collection_day = d + (q - rem + r)
        
        ans.append(str(collection_day))
    
    sys.stdout.write("
".join(ans) + "
")

if __name__ == '__main__':
    main()