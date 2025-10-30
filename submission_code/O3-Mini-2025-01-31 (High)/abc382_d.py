def main():
    import sys
    data = sys.stdin.read().split()
    if not data:
        return
    N = int(data[0])
    M = int(data[1])
    
    result = []
    
    def dfs(pos, current):
        # Base case: when we have a complete sequence of length N
        if pos == N:
            result.append(current.copy())
            return
        
        # If we're starting the sequence, A1 must be at least 1.
        # Otherwise, the next element must be at least the previous element + 10.
        if pos == 0:
            low = 1
        else:
            low = current[-1] + 10
        
        # For the remaining (N - pos) elements, even if we add the minimum gap of 10 for each subsequent element,
        # the largest value for the current element x must satisfy:
        # x + 10*(remaining positions) <= M   -->   x <= M - 10 * (remaining positions)
        remaining = N - pos - 1
        high = M - 10 * remaining
        
        for x in range(low, high + 1):
            current.append(x)
            dfs(pos + 1, current)
            current.pop()
    
    dfs(0, [])
    
    output_lines = [str(len(result))]
    for seq in result:
        output_lines.append(" ".join(map(str, seq)))
    sys.stdout.write("
".join(output_lines))

if __name__ == "__main__":
    main()