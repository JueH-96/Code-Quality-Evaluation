def solve():
    import sys
    input_data = sys.stdin.read().strip().split()
    N, Q = map(int, input_data[:2])
    S = list(input_data[2])
    
    # Precompute occurrences of 'ABC'
    occ = [0]*(N)
    for i in range(N-2):
        if S[i] == 'A' and S[i+1] == 'B' and S[i+2] == 'C':
            occ[i] = 1
    total_occ = sum(occ)
    
    idx = 3  # start index for queries in input_data
    out = []
    
    for _ in range(Q):
        X = int(input_data[idx]); idx += 1
        C = input_data[idx]; idx += 1
        x = X - 1  # convert to 0-based index
        
        if S[x] == C:
            # no change in string
            out.append(str(total_occ))
            continue
        
        # Update character
        S[x] = C
        
        # Only need to re-check positions x-2, x-1, x (if valid)
        for start in (x-2, x-1, x):
            if 0 <= start <= N-3:
                old_val = occ[start]
                if S[start] == 'A' and S[start+1] == 'B' and S[start+2] == 'C':
                    new_val = 1
                else:
                    new_val = 0
                occ[start] = new_val
                total_occ += (new_val - old_val)
        
        out.append(str(total_occ))
    
    print("
".join(out))

def main():
    solve()

if __name__ == "__main__":
    main()