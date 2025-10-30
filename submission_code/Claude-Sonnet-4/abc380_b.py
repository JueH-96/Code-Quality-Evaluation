# YOUR CODE HERE
S = input().strip()

# Find all positions of '|'
pipe_positions = []
for i in range(len(S)):
    if S[i] == '|':
        pipe_positions.append(i)

# Count dashes between consecutive pipes
A = []
for i in range(len(pipe_positions) - 1):
    start = pipe_positions[i]
    end = pipe_positions[i + 1]
    dash_count = end - start - 1  # Number of characters between the pipes
    A.append(dash_count)

# Output the result
print(' '.join(map(str, A)))