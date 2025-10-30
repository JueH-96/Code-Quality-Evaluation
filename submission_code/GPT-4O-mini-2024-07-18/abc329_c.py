def count_repeated_substrings(N, S):
    count = 0
    i = 0
    
    while i < N:
        char = S[i]
        length = 0
        
        # Count the length of the current character sequence
        while i < N and S[i] == char:
            length += 1
            i += 1
        
        # For a sequence of length 'length', the number of non-empty substrings
        # that can be formed is length * (length + 1) // 2
        count += (length * (length + 1)) // 2
    
    return count

import sys
input = sys.stdin.read
data = input().split()
N = int(data[0])
S = data[1]

result = count_repeated_substrings(N, S)
print(result)