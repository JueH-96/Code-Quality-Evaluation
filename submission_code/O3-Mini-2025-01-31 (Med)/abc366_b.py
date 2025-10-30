def main():
    import sys
    data = sys.stdin.read().splitlines()
    if not data:
        return
    n = int(data[0].strip())
    # S_1, S_2, …, S_n are given in order.
    # According to the problem conditions, S_1's letters should appear on the Nth (top) row,
    # S_2's letters on the (N-1)th row, …, and S_n's letters on the 1st (bottom) row.
    S = data[1:1+n]
    # Reverse S so that S_rev[0] will be placed at the bottom,
    # S_rev[1] at the next, and so on.
    S_rev = S[::-1]
    
    # M is the maximum string length.
    M = max(len(s) for s in S)
    
    # We will build M vertical strings (columns) T_1, T_2, …, T_M.
    # In each column j (0-indexed), the character at index r (r from 0 bottom to n-1 top)
    # should be the j-th letter of S_rev[r] if it exists; otherwise, we fill with '*'.
    # After building a column, we trim any trailing '*' (i.e. the top end) so that
    # the string does not end with '*'.
    result = []
    for j in range(M):
        col_chars = []
        # r corresponds to the row where row0 is the bottom.
        for s in S_rev:
            if j < len(s):
                col_chars.append(s[j])
            else:
                col_chars.append('*')
        # Convert list of characters into a string.
        col_str = "".join(col_chars)
        # Remove trailing '*' characters, ensuring the top-most character is not '*'.
        col_str = col_str.rstrip('*')
        result.append(col_str)
    
    sys.stdout.write("
".join(result))

if __name__ == '__main__':
    main()