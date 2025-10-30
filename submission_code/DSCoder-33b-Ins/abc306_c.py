# YOUR CODE HERE

N = int(input())
A = list(map(int, input().split()))

# Create a dictionary to store the indices of each number
indices = {}
for i in range(3*N):
    if A[i] not in indices:
        indices[A[i]] = [i+1]
    else:
        indices[A[i]].append(i+1)

# Sort the numbers based on the middle index
numbers = sorted(indices.keys(), key=lambda x: indices[x][1])

print(*numbers)