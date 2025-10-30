def main():
    import sys
    input_data = sys.stdin.read().split()
    if not input_data:
        return

    it = iter(input_data)
    n = int(next(it))
    q = int(next(it))
    
    # Convert the string S into a list for easier updates.
    S = list(next(it).strip())
    
    # Function to check if "ABC" appears starting at index i.
    def is_ABC(i):
        return i >= 0 and i + 2 < n and S[i] == 'A' and S[i + 1] == 'B' and S[i + 2] == 'C'
    
    # Initialize count of "ABC" substrings in S.
    cnt = 0
    for i in range(n - 2):
        if S[i] == 'A' and S[i + 1] == 'B' and S[i + 2] == 'C':
            cnt += 1

    result = []
    # Process each query.
    for _ in range(q):
        pos = int(next(it)) - 1  # 0-indexed position.
        new_char = next(it)
        
        # The substring "ABC" could be affected if its starting index is pos-2, pos-1 or pos.
        for i in (pos - 2, pos - 1, pos):
            if i >= 0 and i + 2 < n and S[i] == 'A' and S[i + 1] == 'B' and S[i + 2] == 'C':
                cnt -= 1
        
        # Perform the update.
        S[pos] = new_char
        
        # After the update, recheck the possibly affected positions.
        for i in (pos - 2, pos - 1, pos):
            if i >= 0 and i + 2 < n and S[i] == 'A' and S[i + 1] == 'B' and S[i + 2] == 'C':
                cnt += 1
        
        result.append(str(cnt))

    # Write the result to stdout.
    sys.stdout.write("
".join(result))


if __name__ == '__main__':
    main()