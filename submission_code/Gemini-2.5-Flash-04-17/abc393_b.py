# YOUR CODE HERE
import sys

# Read the input string S from standard input
S = sys.stdin.readline().strip()

# Get the length of the string
n = len(S)

# Initialize a counter to store the number of valid triples (i, j, k)
count = 0

# We are looking for triples of integers (i, j, k) that satisfy the conditions:
# 1 <= i < j < k <= |S| (using 1-based indexing as in the problem)
# j - i = k - j
# S_i = 'A', S_j = 'B', S_k = 'C'

# Converting to 0-based indexing (which Python uses):
# Let i' = i - 1, j' = j - 1, k' = k - 1.
# The conditions become:
# 0 <= i' < j' < k' < n (where n = |S|)
# (j' + 1) - (i' + 1) = (k' + 1) - (j' + 1)  =>  j' - i' = k' - j'
# S[i'] = 'A', S[j'] = 'B', S[k'] = 'C'

# The condition j' - i' = k' - j' implies that the indices i', j', k' form an arithmetic progression.
# Let d = j' - i'. Since i' < j', d must be a positive integer (d >= 1).
# The condition k' - j' = d implies k' = j' + d.
# So the triple of 0-based indices must be of the form (j' - d, j', j' + d) for some d >= 1.
# We need to ensure these indices are within the valid range [0, n-1]:
# The first index i' = j' - d must be >= 0  =>  d <= j'.
# The third index k' = j' + d must be < n   =>  d < n - j'  =>  d <= n - 1 - j'.
# Also, we require d >= 1.
# Combining these, the common difference d must be in the range [1, min(j', n - 1 - j')].

# We can iterate through all possible positions for the 'B' character (index j' in 0-based)
# The index j' can range from 0 to n-1. The constraints on d will naturally handle the boundaries
# (i.e., if j' is too close to 0 or n-1, the range of d will be empty).
for j_prime in range(n):
    # If the character at the current index j' is not 'B', it cannot be the middle element. Skip it.
    if S[j_prime] == 'B':
        # If S[j'] is 'B', we look for 'A' at j'-d and 'C' at j'+d for d >= 1.
        
        # Calculate the maximum possible value for the common difference d
        # based on the constraints derived above: 1 <= d <= min(j', n - 1 - j')
        max_d = min(j_prime, n - 1 - j_prime)
        
        # Iterate through all possible positive common differences d
        # The range for d is [1, max_d]. The range() function is exclusive of the stop value,
        # so we use max_d + 1 to include max_d.
        for d in range(1, max_d + 1):
            # Calculate the corresponding indices i' and k' for the current d
            i_prime = j_prime - d
            k_prime = j_prime + d
            
            # Check if the characters at the calculated indices i' and k' are 'A' and 'C' respectively.
            # Because of how max_d was calculated, we are guaranteed that 0 <= i_prime < j_prime < k_prime < n.
            if S[i_prime] == 'A' and S[k_prime] == 'C':
                # If both character conditions are met, we have found a valid triple (i', j', k')
                # which corresponds to a valid triple (i, j, k) in 1-based indexing.
                count += 1

# Print the final count of valid triples
print(count)