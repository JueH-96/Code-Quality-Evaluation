from collections import defaultdict

N = int(input())
A = list(map(int, input().split()))

# Create a dictionary to store the last two positions of each element
positions = defaultdict(lambda: [-1, -1])
for i, a in enumerate(A):
    # Update the positions of the current element
    positions[a][0], positions[a][1] = positions[a][1], i

# Initialize the answer
answer = 0

# Iterate over the elements of A in reverse order
for i in range(N - 1, -1, -1):
    a = A[i]
    # Get the last two positions of the current element
    pos1, pos2 = positions[a]
    # Calculate the contribution of the current element to the answer
    contribution = (i - pos1) * (N - i) + (pos1 - pos2)
    # Add the contribution to the answer
    answer += contribution
    # Update the positions of the current element
    positions[a][0], positions[a][1] = i, positions[a][0]

print(answer)