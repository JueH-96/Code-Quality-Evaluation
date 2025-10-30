import sys

def find_shortest_palindrome(s):
    # Reverse the input string
    rev_s = s[::-1]
    
    # Concatenate the original string and its reverse
    combined = s + '#' + rev_s
    
    # Compute the longest palindromic prefix using KMP algorithm
    lps = [0] * len(combined)
    j = 0
    for i in range(1, len(combined)):
        while j > 0 and combined[i] != combined[j]:
            j = lps[j-1]
        if combined[i] == combined[j]:
            j += 1
        lps[i] = j
    
    # Construct the shortest palindrome
    return rev_s[:len(s)-lps[-1]] + s

# Read input from stdin
s = input().strip()

# Find the shortest palindrome
result = find_shortest_palindrome(s)

# Write output to stdout
print(result)