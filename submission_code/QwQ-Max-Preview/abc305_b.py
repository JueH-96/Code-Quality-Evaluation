p, q = input().split()

positions = {'A': 0}
current = 0
# Define the order of points and their respective distances from the previous point
steps = [('B', 3), ('C', 1), ('D', 4), ('E', 1), ('F', 5), ('G', 9)]
for point, dist in steps:
    current += dist
    positions[point] = current

distance = abs(positions[p] - positions[q])
print(distance)