def shortest_palindrome(S):
    # Create a new string that is S + '#' + reversed(S)
    # The '#' is a separator that cannot be in S
    combined = S + '#' + S[::-1]
    
    # Compute the KMP table (prefix function)
    n = len(combined)
    lps = [0] * n  # Longest Prefix Suffix array
    j = 0  # length of previous longest prefix suffix

    # Build the LPS array
    for i in range(1, n):
        while (j > 0 and combined[i] != combined[j]):
            j = lps[j - 1]
        if combined[i] == combined[j]:
            j += 1
            lps[i] = j
        else:
            lps[i] = 0

    # The length of the longest palindromic suffix
    longest_palindromic_suffix_length = lps[-1]
    
    # The part of S that needs to be added to make it a palindrome
    to_add = S[longest_palindromic_suffix_length:][::-1]
    
    # Construct the shortest palindrome
    return S + to_add

import sys

# Read input
input_string = sys.stdin.read().strip()

# Get the result
result = shortest_palindrome(input_string)

# Print the result
print(result)