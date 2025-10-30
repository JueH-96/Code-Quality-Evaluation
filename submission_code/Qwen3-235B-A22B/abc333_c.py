import itertools

def main():
    import sys
    n = int(sys.stdin.readline())
    
    # Generate repunits up to 100 digits long to cover large possible sums
    repunits = []
    for k in range(1, 101):
        r = (10**k - 1) // 9
        repunits.append(r)
    
    sums = set()
    # Generate all possible combinations of three repunits (allowing repeats)
    for triplet in itertools.combinations_with_replacement(repunits, 3):
        s = sum(triplet)
        sums.add(s)
    
    # Sort the unique sums
    sorted_sums = sorted(sums)
    
    # Output the N-th smallest sum
    print(sorted_sums[n-1])

if __name__ == "__main__":
    main()