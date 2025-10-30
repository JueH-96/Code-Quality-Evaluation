# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

count = 0

for color in range(1, n + 1):
    # Find the two positions of this color
    positions = []
    for i in range(2 * n):
        if a[i] == color:
            positions.append(i)
    
    # Check if there's exactly one person between the two positions
    pos1, pos2 = positions[0], positions[1]
    if pos2 - pos1 == 2:
        count += 1

print(count)