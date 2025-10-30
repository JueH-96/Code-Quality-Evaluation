def main():
    import sys, math
    data = sys.stdin.read().split()
    if not data:
        return
    D = int(data[0])
    
    best = 10**18  # a large number, larger than any possible difference
    # x^2 is at most D so x can go from 0 to floor(sqrt(D))
    max_x = math.isqrt(D)
    for x in range(max_x + 1):
        rem = D - x * x
        # For a given x, we want y such that y^2 is as close as possible to rem.
        y = math.isqrt(rem)
        # Check candidate y (which is floor(sqrt(rem)))
        sum1 = x * x + y * y
        best = min(best, abs(sum1 - D))
        if best == 0:  # Early exit if perfect representation found
            print(0)
            return
        # Check candidate y+1
        sum2 = x * x + (y + 1) * (y + 1)
        best = min(best, abs(sum2 - D))
        if best == 0:
            print(0)
            return
            
    print(best)

if __name__ == '__main__':
    main()