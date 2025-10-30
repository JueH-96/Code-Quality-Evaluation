def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    # Read input
    idx = 0
    N = int(input_data[idx]); idx += 1
    A = list(map(int, input_data[idx:idx+N]))
    idx += N
    
    M = int(input_data[idx]); idx += 1
    B = list(map(int, input_data[idx:idx+M]))
    idx += M
    
    L = int(input_data[idx]); idx += 1
    C = list(map(int, input_data[idx:idx+L]))
    idx += L
    
    Q = int(input_data[idx]); idx += 1
    X = list(map(int, input_data[idx:idx+Q]))
    
    # Precompute all possible sums of one element from B and one from C
    possible_bc_sums = set()
    for b in B:
        for c in C:
            possible_bc_sums.add(b + c)
    
    # For each query, check if there's a sum A_i + bc_sum = X[i]
    # bc_sum = X[i] - A_i
    # We do an O(N) check per query; N <= 100, Q up to 2*10^5 -> at most 2*10^7 checks
    # which can be borderline in Python, but we'll implement it as is.
    
    out = []
    for x_val in X:
        found = False
        for a in A:
            if x_val - a in possible_bc_sums:
                out.append("Yes")
                found = True
                break
        if not found:
            out.append("No")
    
    # Print results
    print("
".join(out))

def main():
    solve()

if __name__ == "__main__":
    main()