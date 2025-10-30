# YOUR CODE HERE
S = input().strip()

# Split by '|' and remove empty strings
segments = S.split('|')

# The first segment will be empty since S starts with '|'
# We need to count dashes in each non-empty segment
A = []
for segment in segments:
    if segment:  # non-empty segment
        A.append(len(segment))

# Print the result
print(' '.join(map(str, A)))