def main():
    import sys
    N = int(sys.stdin.readline())
    # Precompute repunits R[k] = integer with k+1 ones, for k=0..L-1
    L = 100
    R = [0] * L
    v = 0
    for i in range(L):
        v = v * 10 + 1
        R[i] = v
    # Generate all sums of three (not necessarily distinct) repunits:
    # Use i <= j <= k to avoid duplicates
    sums = set()
    for i in range(L):
        ri = R[i]
        for j in range(i, L):
            rj = R[j]
            sij = ri + rj
            for k in range(j, L):
                sums.add(sij + R[k])
    # Sort the unique sums and output the N-th smallest
    ans = sorted(sums)[N-1]
    print(ans)

if __name__ == "__main__":
    main()