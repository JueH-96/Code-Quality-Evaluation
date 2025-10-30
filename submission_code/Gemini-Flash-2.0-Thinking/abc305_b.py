p, q = input().split()

points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
adjacent_distances = [3, 1, 4, 1, 5, 9]

positions = {}
current_position = 0
positions['A'] = current_position
for i in range(len(points) - 1):
    current_position += adjacent_distances[i]
    positions[points[i+1]] = current_position

distance = abs(positions[p] - positions[q])
print(distance)