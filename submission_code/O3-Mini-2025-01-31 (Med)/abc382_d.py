def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    
    results = []
    
    def dfs(seq, idx):
        # If we have filled N values, add the sequence to results.
        if idx == N:
            results.append(seq[:])
            return
        # Determine the range for the next element.
        if idx == 0:
            # For the first element, A1 is at least 1
            # and must allow for N-1 increments of at least 10,
            # so A1 <= M - 10*(N-1)
            start_val = 1
            end_val = M - 10 * (N - 1)
        else:
            # For subsequent elements, A[i] must be at least A[i-1]+10
            start_val = seq[-1] + 10
            # And it must also allow for remaining positions to reach at most M.
            # If there are (N - idx - 1) positions remaining, then:
            end_val = M - 10 * (N - idx - 1)
            
        for candidate in range(start_val, end_val + 1):
            dfs(seq + [candidate], idx + 1)
    
    dfs([], 0)
    
    # First print the count of sequences, and then each sequence on a new line.
    out_lines = []
    out_lines.append(str(len(results)))
    for seq in results:
        out_lines.append(" ".join(map(str, seq)))
        
    sys.stdout.write("
".join(out_lines))
    
if __name__ == '__main__':
    main()