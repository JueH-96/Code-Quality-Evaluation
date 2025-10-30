def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    
    n = int(data[0])
    A = list(map(int, data[1:]))
    total = sum(A)
    
    # Let t be the base value that we want to hit: t = S//n.
    # Since the operations conserve the sum, the only possible nearly-equal
    # distribution is to have (n - rem) elements equal to t and rem elements equal to t+1,
    # where rem = total % n.
    t = total // n
    rem = total % n  # Number of elements which should be t+1.
    
    A.sort()
    
    # We want to assign the smallest n-rem elements to target value t
    # and the largest rem elements to target value t+1.
    # For each element that is above its target, we must remove the surplus.
    # Each operation transfers one unit from a surplus element to a deficit element,
    # so the minimal number of operations equals the total surplus.
    ops = 0
    # For the first (n-rem) elements, desired value is t.
    for i in range(n - rem):
        if A[i] > t:
            ops += A[i] - t
    # For the last rem elements, desired value is t+1.
    for i in range(n - rem, n):
        if A[i] > t + 1:
            ops += A[i] - (t + 1)
    
    sys.stdout.write(str(ops))
    
if __name__ == '__main__':
    main()