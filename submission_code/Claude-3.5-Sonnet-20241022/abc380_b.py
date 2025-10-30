# Read input string
S = input().strip()

# Initialize variables
result = []
count = 0

# Process the string character by character
for i in range(1, len(S)):
    if S[i] == '-':
        count += 1
    elif S[i] == '|':
        result.append(count)
        count = 0

# Print the result
print(' '.join(map(str, result)))