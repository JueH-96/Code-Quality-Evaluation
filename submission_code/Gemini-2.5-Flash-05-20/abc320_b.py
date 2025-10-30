import sys

# Read the input string S from standard input
S = sys.stdin.readline().strip()

# Get the length of the string S
N = len(S)

# Initialize max_palindrome_length to 0.
# Since the problem states S has a length between 2 and 100,
# and any single character is a palindrome, the maximum length will always be at least 1.
# Initializing to 0 is fine as the first single-character palindrome found will update it to 1.
max_palindrome_length = 0

# Iterate through all possible starting positions of a substring
for i in range(N):
    # Iterate through all possible ending positions of a substring
    # The substring S[i:j+1] starts at index i and ends at index j (inclusive)
    for j in range(i, N):
        # Extract the current contiguous substring
        current_substring = S[i : j + 1]

        # Check if the current_substring is a palindrome
        # A string is a palindrome if it reads the same forwards and backwards.
        # Python's slicing `[::-1]` creates a reversed copy of the string.
        if current_substring == current_substring[::-1]:
            # If it is a palindrome, update the maximum length found so far
            max_palindrome_length = max(max_palindrome_length, len(current_substring))

# Print the final maximum palindrome length
print(max_palindrome_length)