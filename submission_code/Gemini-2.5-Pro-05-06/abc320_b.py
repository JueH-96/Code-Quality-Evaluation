# YOUR CODE HERE
S = input()
n = len(S)
max_pal_length = 0

# Constraints state S has length between 2 and 100.
# Thus, n >= 2.
# The loops below will find single character palindromes (e.g., S[0:1]),
# so max_pal_length will become at least 1.
# Initializing max_pal_length to 0 is therefore fine.

# Iterate over all possible starting positions of a substring
for i in range(n):
    # Iterate over all possible ending positions of a substring
    # The substring S[i...j] (inclusive) corresponds to Python slice S[i : j+1]
    for j in range(i, n):
        substring = S[i : j+1]
        
        # Check if the current substring is a palindrome
        # A string is a palindrome if it reads the same forwards and backwards.
        if substring == substring[::-1]:
            # If it's a palindrome, its length is len(substring).
            # Update max_pal_length if this palindrome is longer.
            if len(substring) > max_pal_length:
                max_pal_length = len(substring)
            # A more concise way to update:
            # max_pal_length = max(max_pal_length, len(substring))
            
# Print the final maximum length found
print(max_pal_length)