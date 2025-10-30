# YOUR CODE HERE
points = {
    'A': 0,
    'B': 3,
    'C': 3 + 1,
    'D': 3 + 1 + 4,
    'E': 3 + 1 + 4 + 1,
    'F': 3 + 1 + 4 + 1 + 5,
    'G': 3 + 1 + 4 + 1 + 5 + 9
}

p, q = input().split()

distance = abs(points[p] - points[q])

print(distance)