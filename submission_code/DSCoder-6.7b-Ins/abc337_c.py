import sys

N = int(sys.stdin.readline().strip())
A = list(map(int, sys.stdin.readline().strip().split()))

# Create a list to store the final order of people
order = [0] * N

# For each person, find their position in the line
for i in range(N):
    if A[i] == -1:
        order[0] = i + 1
    else:
        order[A[i]] = i + 1

# Print the order of people
print(' '.join(map(str, order)))