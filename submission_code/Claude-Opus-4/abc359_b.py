# YOUR CODE HERE
N = int(input())
A = list(map(int, input().split()))

count = 0

# For each color from 1 to N
for color in range(1, N + 1):
    # Find the two positions where this color appears
    positions = []
    for i in range(2 * N):
        if A[i] == color:
            positions.append(i)
    
    # Check if there's exactly one person between them
    # If positions are at index i and j, there are (j - i - 1) people between them
    if len(positions) == 2:
        if positions[1] - positions[0] == 2:
            count += 1

print(count)