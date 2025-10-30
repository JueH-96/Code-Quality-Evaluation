def main():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    n = int(data[0])
    A = list(map(int, data[1:]))
    
    total = sum(A)
    # Determine the ideal final values:
    # Let q = total // n and r = total % n.
    # We want (n-r) elements to be q and r elements to be q+1.
    q = total // n
    r = total % n
    
    # The idea is that in each allowed operation we transfer one unit from one entry 
    # to another. Thus the number of operations required is exactly the total number 
    # of unit adjustments (either from deficits or surpluses) that must be made.
    # By redistributing optimally, we can choose to assign the lower target (q) to the 
    # smallest values and q+1 to the largest values.
    #
    # For an element with value X assigned a target T, if X < T we must add (T - X) units.
    # Summing over all elements gives the minimal number of unit moves (and thus operations)
    # needed because every operation moves exactly one unit.
    
    A.sort()
    ops = 0
    # Assign the (n-r) smallest to have final value q
    for i in range(n - r):
        if A[i] < q:
            ops += (q - A[i])
    # Assign the last r (largest) to have final value q+1
    for i in range(n - r, n):
        if A[i] < q + 1:
            ops += (q + 1 - A[i])
    
    sys.stdout.write(str(ops))
    
if __name__ == '__main__':
    main()