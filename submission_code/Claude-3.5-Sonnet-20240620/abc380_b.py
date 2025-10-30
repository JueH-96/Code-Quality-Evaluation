# YOUR CODE HERE
def reconstruct_sequence(S):
    A = []
    count = 0
    for char in S[1:]:  # Skip the first '|'
        if char == '-':
            count += 1
        elif char == '|':
            A.append(count)
            count = 0
    return A

# Read input
S = input().strip()

# Reconstruct the sequence
result = reconstruct_sequence(S)

# Print the result
print(' '.join(map(str, result)))