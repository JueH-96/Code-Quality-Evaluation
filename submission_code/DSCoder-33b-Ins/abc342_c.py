# YOUR CODE HERE

# Read the inputs
N = int(input())
S = input()
Q = int(input())

# Initialize a dictionary to store the replacements
replacements = {}

# Read the replacements
for _ in range(Q):
    c, d = input().split()
    replacements[c] = d

# Replace the characters in S
for i in range(N):
    if S[i] in replacements:
        S = S[:i] + replacements[S[i]] + S[i+1:]

# Print the result
print(S)