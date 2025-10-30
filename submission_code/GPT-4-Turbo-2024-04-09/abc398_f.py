import sys
input = sys.stdin.read

def shortest_palindrome(s):
    n = len(s)
    rev_s = s[::-1]
    # We create a new string which is s + '#' + reversed(s)
    # '#' acts as a separator that doesn't match any character in s
    combined = s + '#' + rev_s
    
    # We will use the KMP (Knuth-Morris-Pratt) algorithm's partial match table to find the palindrome
    # This table will help us determine the longest suffix of the reversed string that matches the prefix of the original string
    kmp_table = [0] * len(combined)
    j = 0
    
    # Build the KMP table
    for i in range(1, len(combined)):
        while (j > 0 and combined[i] != combined[j]):
            j = kmp_table[j - 1]
        if combined[i] == combined[j]:
            j += 1
            kmp_table[i] = j
    
    # The last value in the KMP table will tell us the maximum length of the suffix of rev_s that matches the prefix of s
    max_len = kmp_table[-1]
    
    # To form the shortest palindrome, we take the original string s and append the suffix of rev_s that hasn't matched
    shortest_pal = s + rev_s[max_len:]
    return shortest_pal

# Read input
s = input().strip()

# Get the shortest palindrome
result = shortest_palindrome(s)

# Print the result
print(result)