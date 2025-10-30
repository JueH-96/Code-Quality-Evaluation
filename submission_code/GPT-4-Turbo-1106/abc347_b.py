# Read the input string from standard input
S = input().strip()

# Initialize a set to store unique substrings
unique_substrings = set()

# Generate all possible non-empty substrings and add them to the set
for i in range(len(S)):
    for j in range(i + 1, len(S) + 1):
        unique_substrings.add(S[i:j])

# The number of unique substrings is the size of the set
print(len(unique_substrings))