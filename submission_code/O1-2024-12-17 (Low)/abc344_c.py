def main():
    import sys
    data = sys.stdin.read().strip().split()
    
    # Read input
    # N
    idx = 0
    N = int(data[idx]); idx += 1
    # A
    A = list(map(int, data[idx:idx+N]))
    idx += N
    # M
    M = int(data[idx]); idx += 1
    # B
    B = list(map(int, data[idx:idx+M]))
    idx += M
    # L
    L = int(data[idx]); idx += 1
    # C
    C = list(map(int, data[idx:idx+L]))
    idx += L
    # Q
    Q = int(data[idx]); idx += 1
    # X
    X = list(map(int, data[idx:idx+Q]))
    idx += Q
    
    # Precompute all possible sums of A and B
    AB_sums = set()
    for a in A:
        for b in B:
            AB_sums.add(a + b)

    # For each X_i, check if it's possible to find a c in C
    # such that (X_i - c) is in AB_sums
    output = []
    for x_val in X:
        # We'll try each c in C to see if (x_val - c) is in AB_sums
        possible = False
        for c in C:
            if (x_val - c) in AB_sums:
                possible = True
                break
        output.append("Yes" if possible else "No")
    
    print("
".join(output))

# Don't forget to call main
if __name__ == "__main__":
    main()