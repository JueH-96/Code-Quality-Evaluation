# YOUR CODE HERE
def shortest_palindrome_with_prefix(S):
    # Reverse the string S
    R = S[::-1]
    
    # Concatenate S and R with a special character in between
    # to avoid overlap issues
    combined = S + '#' + R
    
    # Compute the KMP table (prefix function)
    n = len(combined)
    pi = [0] * n
    
    # KMP prefix function calculation
    for i in range(1, n):
        j = pi[i - 1]
        while j > 0 and combined[i] != combined[j]:
            j = pi[j - 1]
        if combined[i] == combined[j]:
            j += 1
        pi[i] = j
    
    # The length of the longest prefix of R that matches a suffix of S
    longest_prefix_suffix = pi[-1]
    
    # Append the necessary part of R to S
    result = S + R[longest_prefix_suffix:]
    
    return result

# Read input
import sys
input = sys.stdin.read().strip()

# Compute and print the result
print(shortest_palindrome_with_prefix(input))