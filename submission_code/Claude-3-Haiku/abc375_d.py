from collections import defaultdict

def count_palindrome_triples(s):
    n = len(s)
    count = 0
    
    # Create a dictionary to store the indices of each character
    char_indices = defaultdict(list)
    for i, c in enumerate(s):
        char_indices[c].append(i)
    
    # Iterate through all possible triples (i, j, k)
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                # Check if the length-3 string formed by concatenating s[i], s[j], and s[k] is a palindrome
                if s[i] == s[k]:
                    count += 1
    
    return count

# Read input from stdin
s = input().strip()

# Solve the problem and write the answer to stdout
print(count_palindrome_triples(s))