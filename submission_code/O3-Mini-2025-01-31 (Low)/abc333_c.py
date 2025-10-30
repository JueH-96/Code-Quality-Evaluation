def main():
    import sys

    # Read the input number N (1-indexed position in the sorted set of sums)
    N = int(sys.stdin.readline().strip())

    # Generate a list of repunits.
    # A repunit of length k in decimal is (10^k - 1) // 9.
    # We choose an upper bound for k such that the number of distinct sums is enough.
    # Through combinatorial calculation, with k from 1 up to 15, there are
    # "combinations with repetition" of 3 repunits from 15 which is C(15+3-1, 3) = C(17,3)=680.
    # That is more than enough for N <= 333.
    repunits = []
    for k in range(1, 16):
        repunit = (10**k - 1) // 9
        repunits.append(repunit)

    # Use a set to store all possible sums of exactly three repunits.
    sums = set()
    L = len(repunits)
    for i in range(L):
        for j in range(i, L):  # allow repeated repunits; using non-decreasing order avoids duplicates
            for k in range(j, L):
                total = repunits[i] + repunits[j] + repunits[k]
                sums.add(total)
    
    # Convert the set to a sorted list to get ascending order.
    sorted_sums = sorted(sums)
    
    # Print the N-th smallest sum. (N is 1-indexed.)
    sys.stdout.write(str(sorted_sums[N-1]))
    
    
if __name__ == '__main__':
    main()