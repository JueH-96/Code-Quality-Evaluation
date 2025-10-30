def main():
    import sys
    N = int(sys.stdin.read())
    
    # Generate repunits up to 100 digits
    repunits = []
    current = 1
    for _ in range(100):
        repunits.append(current)
        current = current * 10 + 1
    
    # Generate all possible sums of three repunits
    sums = set()
    for a in repunits:
        for b in repunits:
            for c in repunits:
                s = a + b + c
                sums.add(s)
    
    # Sort the sums
    sorted_sums = sorted(sums)
    
    # Output the N-th element (1-based index)
    print(sorted_sums[N-1])

if __name__ == "__main__":
    main()