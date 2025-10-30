# Read the input
N = int(input())
S = input()

# Initialize the number of operations to 0
operations = 0

# Find the first and last index of 1s
first_one = -1
last_one = -1
for i in range(N):
    if S[i] == '1':
        if first_one == -1:
            first_one = i
        last_one = i

# Swap the characters to make all 1s contiguous
for i in range(first_one, last_one):
    if S[i] == '0' and S[i+1] == '1':
        S = S[:i] + S[i+1] + S[i] + S[i+2:]
        operations += 1

# Print the answer
print(operations)