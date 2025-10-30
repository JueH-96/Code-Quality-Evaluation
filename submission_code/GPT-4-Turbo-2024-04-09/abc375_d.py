def count_palindromic_triples(S):
    n = len(S)
    count = 0
    
    # We need to count the number of valid (i, j, k) such that S[i] == S[k]
    # We can use a frequency count array to optimize the counting process
    
    # freq[c][j] will store the number of times character c appears at indices >= j
    from collections import defaultdict
    import sys
    input = sys.stdin.read
    S = input().strip()
    
    # Initialize frequency dictionary for each character
    freq = defaultdict(lambda: [0] * (n + 1))
    
    # Fill frequency from the end of the string
    for i in range(n - 1, -1, -1):
        for c in range(26):
            char = chr(c + ord('A'))
            freq[char][i] = freq[char][i + 1]
        freq[S[i]][i] += 1
    
    # Now calculate the number of valid triples
    for j in range(1, n - 1):
        for i in range(j):
            if S[i] == S[j - 1]:  # S[i] should be equal to S[k] where k > j
                count += freq[S[i]][j + 1]  # All k > j where S[k] == S[i]
    
    print(count)