def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    it = iter(data)
    n = int(next(it))
    A = int(next(it))
    B = int(next(it))
    T = A + B
    # Read D_i values and compute each D_i mod (A+B)
    D = [int(next(it)) for _ in range(n)]
    r = [d % T for d in D]
    r.sort()
    
    # We now have a set of mod values on a circle of length T.
    # The problem becomes: is there a way (a shift of the circle, i.e. choosing today)
    # so that every (d + today shift) mod T falls in [0, A)?
    # This is equivalent to: Do all our r values lie inside an arc of length A on the circle?
    #
    # To check this, we will use the standard technique for covering points on a circle:
    # Duplicate the sorted list with T added to the second copy, then for each starting point,
    # check how many consecutive points lie within r[i] + A.
    
    r2 = r + [x + T for x in r]
    j = 0
    possible = False
    for i in range(n):
        # Move j forward as long as the current point is within r[i] + A.
        while j < len(r2) and r2[j] < r2[i] + A:
            j += 1
        # If the window covers all n points, then an arc of length A covers all points.
        if j - i >= n:
            possible = True
            break
    print("Yes" if possible else "No")
    
if __name__ == '__main__':
    main()