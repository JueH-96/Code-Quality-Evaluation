def count_palindrome_triples(S):
    from collections import defaultdict
    
    n = len(S)
    count = 0
    
    # Dictionary to count occurrences of characters
    char_count = defaultdict(int)
    
    # We will iterate through the string and keep track of counts
    for j in range(n):
        # For each character S[j], we will consider it as the middle character of the palindrome
        # We need to find pairs (i, k) such that S[i] == S[k] == S[j]
        
        # Before processing S[j], we will count how many times S[j] has appeared before
        if j > 0:
            count += char_count[S[j]] * (j - 1)  # (j-1) is the number of valid k's for each i
        
        # Now we will update the count of S[j] for future k's
        char_count[S[j]] += 1
    
    # Now we need to count the pairs (i, k) for each character
    # Resetting the character count for the second pass
    char_count.clear()
    
    # We will iterate again to count valid (i, k) pairs
    for j in range(n):
        # For each character S[j], we will consider it as the middle character of the palindrome
        # We need to find pairs (i, k) such that S[i] == S[k] == S[j]
        
        # Count how many times S[j] has appeared before
        if j > 0:
            count += char_count[S[j]]  # This counts valid i's for each k
        
        # Now we will update the count of S[j] for future i's
        char_count[S[j]] += 1
    
    return count

# Read input
import sys
input = sys.stdin.read
S = input().strip()

# Get the result
result = count_palindrome_triples(S)

# Print the result
print(result)