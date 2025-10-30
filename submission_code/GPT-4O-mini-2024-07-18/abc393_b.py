def count_triples(S):
    n = len(S)
    count = 0
    
    # Iterate through possible j values
    for j in range(1, n - 1):
        # j is the middle index, we need to find i and k
        # i must be less than j and k must be greater than j
        # j - i = k - j => k = 2*j - i
        for i in range(j):
            k = 2 * j - i
            if k < n and S[i] == 'A' and S[j] == 'B' and S[k] == 'C':
                count += 1
                
    return count

# Read input
import sys
input = sys.stdin.read
S = input().strip()

# Get the result and print it
result = count_triples(S)
print(result)