def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    Q = int(data[1])
    S = list(data[2])
    
    queries = []
    index = 3
    for _ in range(Q):
        X_i = int(data[index]) - 1
        C_i = data[index + 1]
        queries.append((X_i, C_i))
        index += 2
    
    # Function to count "ABC" in the string
    def count_abc(s):
        count = 0
        for i in range(len(s) - 2):
            if s[i] == 'A' and s[i + 1] == 'B' and s[i + 2] == 'C':
                count += 1
        return count
    
    # Initial count of "ABC"
    current_count = count_abc(S)
    
    # Process each query
    results = []
    for X_i, C_i in queries:
        # Check the affected region before change
        if X_i > 0 and S[X_i - 1:X_i + 2] == ['A', 'B', 'C']:
            current_count -= 1
        if X_i < N - 2 and S[X_i:X_i + 3] == ['A', 'B', 'C']:
            current_count -= 1
        
        # Perform the change
        S[X_i] = C_i
        
        # Check the affected region after change
        if X_i > 0 and S[X_i - 1:X_i + 2] == ['A', 'B', 'C']:
            current_count += 1
        if X_i < N - 2 and S[X_i:X_i + 3] == ['A', 'B', 'C']:
            current_count += 1
        
        # Append the result after this query
        results.append(current_count)
    
    # Output all results
    sys.stdout.write('
'.join(map(str, results)) + '
')

if __name__ == "__main__":
    main()