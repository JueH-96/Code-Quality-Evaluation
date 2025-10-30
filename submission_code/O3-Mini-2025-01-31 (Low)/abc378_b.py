def main():
    import sys
    input = sys.stdin.readline
    
    # Read number of garbage types
    N = int(input().strip())
    schedules = [None] * N
    for i in range(N):
        q, r = map(int, input().split())
        schedules[i] = (q, r)
    
    # Read number of queries
    Q = int(input().strip())
    for _ in range(Q):
        t, d = map(int, input().split())
        # Adjust for 0-indexing
        q, r = schedules[t - 1]
        
        # Compute the next collection day:
        # If d mod q is equal to r then day d itself is a collection day.
        # Otherwise, add the smallest positive offset to reach a day where day mod q == r.
        mod_d = d % q
        if mod_d == r:
            day = d
        else:
            # Compute the difference (taking modulo arithmetic into account)
            # (r - mod_d) might be negative so we add q and reduce modulo q.
            diff = (r - mod_d + q) % q
            day = d + diff
        sys.stdout.write(str(day) + "
")
        
if __name__ == '__main__':
    main()