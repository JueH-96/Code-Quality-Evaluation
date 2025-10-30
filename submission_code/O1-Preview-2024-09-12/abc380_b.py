# YOUR CODE HERE
S = input()
segments = S.strip().split('|')

# Remove the first and last elements if they're empty strings
if segments and segments[0] == '':
    segments = segments[1:]
if segments and segments[-1] == '':
    segments = segments[:-1]

# Calculate the lengths of the segments which correspond to A_i
A = [len(segment) for segment in segments]

# Output the sequence A
print(' '.join(map(str, A)))