def main():
    import sys
    input_data = sys.stdin.read().split()
    n = int(input_data[0])
    q = int(input_data[1])
    s = list(input_data[2])
    
    # Precompute the initial occurrences of "ABC".
    count = 0
    valid = [False] * n  # valid[i] indicates if s[i:i+3] equals "ABC"
    for i in range(n - 2):
        if s[i] == 'A' and s[i + 1] == 'B' and s[i + 2] == 'C':
            valid[i] = True
            count += 1

    results = []
    index = 3
    for _ in range(q):
        # Parse query input.
        pos = int(input_data[index]) - 1  # Convert to 0-based index.
        ch = input_data[index + 1]
        index += 2
        
        # The update can affect any occurrence that starts at one of these indices.
        left_bound = max(0, pos - 2)
        right_bound = min(n - 3, pos)  # Starting indices for substring of length 3
        
        # Remove the count from affected windows before updating.
        for i in range(left_bound, right_bound + 1):
            if valid[i]:
                count -= 1
                valid[i] = False
        
        # Update the character at the specified position.
        s[pos] = ch
        
        # Recheck and update possible substrings that might now form "ABC".
        for i in range(left_bound, right_bound + 1):
            if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
                if not valid[i]:
                    valid[i] = True
                    count += 1
        
        # Append the result for this query.
        results.append(str(count))
    
    sys.stdout.write("
".join(results))

if __name__ == '__main__':
    main()