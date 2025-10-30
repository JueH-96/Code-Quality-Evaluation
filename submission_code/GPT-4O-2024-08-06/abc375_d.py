# YOUR CODE HERE
def count_palindrome_triples(S):
    n = len(S)
    count = {}
    total_triples = 0
    
    for k in range(n):
        char_k = S[k]
        
        if char_k in count:
            # If char_k has appeared before, calculate possible triples
            total_triples += count[char_k] * (k - count[char_k])
        
        # Update the count of char_k
        if char_k in count:
            count[char_k] += 1
        else:
            count[char_k] = 1
    
    return total_triples

import sys
input = sys.stdin.read
S = input().strip()
print(count_palindrome_triples(S))