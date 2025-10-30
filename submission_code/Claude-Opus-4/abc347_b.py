# YOUR CODE HERE
S = input().strip()

# Use a set to store unique substrings
unique_substrings = set()

# Generate all possible substrings
n = len(S)
for i in range(n):
    for j in range(i + 1, n + 1):
        substring = S[i:j]
        unique_substrings.add(substring)

# Print the count of unique substrings
print(len(unique_substrings))