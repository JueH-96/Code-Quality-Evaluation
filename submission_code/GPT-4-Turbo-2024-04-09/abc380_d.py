import sys
input = sys.stdin.read

def solve():
    data = input().split()
    S = data[0]
    Q = int(data[1])
    K = list(map(int, data[2:]))
    
    # Length of the initial string S
    n = len(S)
    
    # We need to determine the character at position K_i in the infinitely expanded string.
    # The pattern repeats every 2*n characters after sufficient expansions.
    # The first n characters are S, the next n characters are T (inverted case of S).
    
    # Precompute the toggled string T
    T = []
    for char in S:
        if char.islower():
            T.append(char.upper())
        else:
            T.append(char.lower())
    T = ''.join(T)
    
    results = []
    for k in K:
        # Adjust k to be zero-indexed
        k -= 1
        # Determine the position in the cycle of length 2*n
        pos = k % (2 * n)
        if pos < n:
            results.append(S[pos])
        else:
            results.append(T[pos - n])
    
    # Print results as a single space-separated line
    print(' '.join(results))