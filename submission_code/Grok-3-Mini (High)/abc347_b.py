# Read the input string from stdin
S = input()

# Get the length of the string
n = len(S)

# Create a set to store unique substrings
substr_set = set()

# Generate all possible substrings and add them to the set
for i in range(n):
    for j in range(i, n):
        substr = S[i:j+1]
        substr_set.add(substr)

# Print the number of unique substrings
print(len(substr_set))