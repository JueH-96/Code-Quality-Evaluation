# Read the positions and calculate the distance
positions = {'A': 0, 'B': 3, 'C': 4, 'D': 8, 'E': 9, 'F': 14, 'G': 23}
p, q = input().split()
distance = abs(positions[p] - positions[q])
print(distance)