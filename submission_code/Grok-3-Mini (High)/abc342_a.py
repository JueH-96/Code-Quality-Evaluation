# YOUR CODE HERE

# Read the input string from stdin
S = input().strip()

# Determine the majority character by examining the first three characters
if S[0] == S[1]:
    majority_char = S[0]
elif S[0] == S[2]:
    majority_char = S[0]
elif S[1] == S[2]:
    majority_char = S[1]

# Iterate through the string to find the position where the character differs
for i in range(len(S)):
    if S[i] != majority_char:
        # Output the 1-based index and break since there's only one such position
        print(i + 1)
        break