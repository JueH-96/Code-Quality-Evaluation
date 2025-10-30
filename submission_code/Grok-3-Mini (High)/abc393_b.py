# Read the input string from stdin
S = input().strip()
n = len(S)
count = 0

# Iterate over each position j in the string
for j in range(n):
    # Check if the character at position j is 'B'
    if S[j] == 'B':
        # Calculate the maximum possible d for this j
        max_d = min(j, n - 1 - j)
        # Iterate over each possible d starting from 1
        for d in range(1, max_d + 1):
            # Check if the characters at j-d and j+d are 'A' and 'C' respectively
            if S[j - d] == 'A' and S[j + d] == 'C':
                count += 1

# Output the count
print(count)