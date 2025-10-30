# Read the input
N, D = map(int, input().split())
S = input()

# Initialize a list to store the result
result = ['.'] * N

# Iterate through the boxes from right to left
for i in range(N-1, -1, -1):
    if S[i] == '@':
        D -= 1
        if D >= 0:
            result[i] = '@'

# Print the result
print(''.join(result))