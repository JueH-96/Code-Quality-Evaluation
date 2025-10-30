# YOUR CODE HERE
from collections import defaultdict

def count_palindrome_triples(S):
    n = len(S)
    count = 0
    char_positions = defaultdict(list)
    
    # Store positions of each character
    for i, char in enumerate(S):
        char_positions[char].append(i)
    
    # Count palindromes
    for j in range(1, n-1):
        for i in char_positions[S[j]]:
            if i >= j:
                break
            for k in char_positions[S[i]]:
                if k <= j:
                    continue
                if k > n-1:
                    break
                count += 1
    
    return count

# Read input
S = input().strip()

# Solve and print output
result = count_palindrome_triples(S)
print(result)