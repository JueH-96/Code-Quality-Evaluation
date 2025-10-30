# YOUR CODE HERE
import sys

# Read the input string
S = sys.stdin.readline().strip()

# Get the length of the string
n = len(S)

# Initialize the counter
count = 0

# Iterate through all possible first indices i (0-based)
# i ranges from 0 to n-1.
for i in range(n):
    # If S[i] is 'A'
    if S[i] == 'A':
        # Iterate through all possible second indices j (0-based)
        # j must be strictly greater than i.
        # j ranges from i + 1 to n-1.
        for j in range(i + 1, n):
            # If S[j] is 'B'
            if S[j] == 'B':
                # Calculate the common difference d = j - i.
                # Since j > i, diff > 0 is guaranteed.
                diff = j - i

                # Calculate the potential third index k.
                # k must be equal to j + diff.
                k = j + diff

                # Check if k is within the string bounds (k < n)
                # and if the character at index k is 'C' (S[k] == 'C').
                # The condition k < n also implies j < k since diff > 0.
                if k < n and S[k] == 'C':
                    # Found a valid triple (i, j, k) that satisfies all conditions.
                    # These are 0-based indices (i, j, k).
                    # The problem asks for count of 1-based indices (i+1, j+1, k+1).
                    # The count is the same.
                    count += 1

# Print the total count
print(count)