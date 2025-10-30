# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

count = 0
for color in range(1, n + 1):
    # Find positions of this color
    positions = []
    for i in range(2 * n):
        if a[i] == color:
            positions.append(i)
    
    # Check if exactly one person between them
    if positions[1] - positions[0] == 2:
        count += 1

print(count)