def main():
    import sys
    # Read the input which is a single integer N.
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    
    # A repunit with i digits is given by (10**i - 1) // 9.
    # We want to represent numbers as the sum of exactly three repunits.
    # Since different triplets (with repetition allowed) may be chosen, we
    # can generate all sums of the form R_i + R_j + R_k for 1 <= i <= j <= k.
    #
    # The number of such triples with repunits having indices 1..L is (L+2 choose 3).
    # For N up to 333, we need to choose an L such that
    #   (L+2)*(L+1)*L/6 >= 333.
    #
    # For L = 11, (13*12*11)/6 = 286 < 333.
    # For L = 12, (14*13*12)/6 = 364 >= 333.
    # So, using L = 12 is sufficient.
    L = 12
    
    # Precompute repunits for digits 1 to L.
    repunits = [(10**i - 1) // 9 for i in range(1, L+1)]
    
    # Generate all sums R_i + R_j + R_k where i<=j<=k
    sums = []
    for i in range(L):
        for j in range(i, L):
            for k in range(j, L):
                sums.append(repunits[i] + repunits[j] + repunits[k])
    
    # Sort the list of candidate sums in ascending order.
    sums.sort()
    
    # The N-th smallest (1-indexed) is at index N-1.
    sys.stdout.write(str(sums[N-1]))
    
if __name__ == '__main__':
    main()