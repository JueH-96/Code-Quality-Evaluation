# YOUR CODE HERE
N, K = map(int, input().split())
S = input()

# Find all groups of consecutive healthy teeth
groups = []
i = 0
while i < N:
    if S[i] == 'O':
        start = i
        while i < N and S[i] == 'O':
            i += 1
        groups.append(i - start)
    else:
        i += 1

# Calculate total strawberries
total = 0
for length in groups:
    total += length // K

print(total)