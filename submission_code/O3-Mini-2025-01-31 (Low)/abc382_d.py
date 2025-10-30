def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return
    N = int(input_data[0])
    M = int(input_data[1])
    
    result = []
    
    def dfs(seq, depth):
        # depth: current length of seq
        if depth == N:
            # complete sequence, store result if last value <= M (should always be true by construction)
            result.append(seq.copy())
            return
        
        # We want the next element. It should be at least (previous value + 10) if seq is non-empty.
        if depth == 0:
            # for first element A1, lower bound is 1.
            low = 1
        else:
            low = seq[-1] + 10
        
        # Compute maximum allowed for the current element
        # Since for the remaining elements we must add at least 10 each time.
        remaining = N - depth
        # Let current element be x, then minimal future sequence is x, x+10, x+20,... x+10*(remaining-1).
        # And final element x+10*(remaining-1) should be <= M.
        # So x <= M - 10*(remaining-1)
        high = M - 10 * (remaining - 1)
        
        for candidate in range(low, high + 1):
            seq.append(candidate)
            dfs(seq, depth + 1)
            seq.pop()
    
    dfs([], 0)
    
    # The sequences are generated in lexicographical order due to the increasing lower bound
    # Print count and then each sequence
    sys.stdout.write(str(len(result)) + "
")
    for seq in result:
        sys.stdout.write(" ".join(map(str, seq)) + "
")
        
if __name__ == '__main__':
    main()