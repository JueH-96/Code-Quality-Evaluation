def main():
    import sys
    
    # Read N
    N = int(sys.stdin.readline().strip())
    
    # Precompute repunits up to length 12. That is certainly enough to cover N<=333.
    repunits = []
    current = 0
    for _ in range(1, 13):
        current = current * 10 + 1
        repunits.append(current)
    
    # Generate all possible sums of exactly three repunits
    possible_sums = set()
    for r1 in repunits:
        for r2 in repunits:
            for r3 in repunits:
                possible_sums.add(r1 + r2 + r3)
    
    # Sort the distinct sums
    sorted_sums = sorted(possible_sums)
    
    # Output the N-th (1-based) smallest such sum
    print(sorted_sums[N-1])

# Don't forget to call main()
main()