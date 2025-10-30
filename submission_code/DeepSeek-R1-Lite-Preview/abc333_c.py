def main():
    import sys
    N = int(sys.stdin.read())
    
    # Generate repunits up to length 12
    max_length = 12
    repunits = [int('1' * k) for k in range(1, max_length + 1)]
    
    # Generate all possible sums of three repunits with a <= b <= c
    sums = []
    for a in range(len(repunits)):
        for b in range(a, len(repunits)):
            for c in range(b, len(repunits)):
                S = repunits[a] + repunits[b] + repunits[c]
                sums.append(S)
    
    # Sort the sums and find the N-th smallest
    sums.sort()
    print(sums[N-1])

if __name__ == "__main__":
    main()