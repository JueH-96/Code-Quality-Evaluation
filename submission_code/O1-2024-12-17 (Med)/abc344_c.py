def main():
    import sys

    data = sys.stdin.read().strip().split()
    # Read input values
    idx = 0
    N = int(data[idx]); idx += 1
    A = list(map(int, data[idx:idx+N]))
    idx += N
    
    M = int(data[idx]); idx += 1
    B = list(map(int, data[idx:idx+M]))
    idx += M
    
    L = int(data[idx]); idx += 1
    C = list(map(int, data[idx:idx+L]))
    idx += L
    
    Q = int(data[idx]); idx += 1
    X = list(map(int, data[idx:idx+Q]))
    idx += Q
    
    # Precompute all sums of B and C
    bc_sums = set()
    for b in B:
        for c in C:
            bc_sums.add(b + c)
    
    # For each query, check if there's a combination that sums to X_i
    out = []
    for x in X:
        possible = False
        for a in A:
            if (x - a) in bc_sums:
                possible = True
                break
        out.append("Yes" if possible else "No")
    
    print("
".join(out))

# Do not forget to call main()
if __name__ == "__main__":
    main()