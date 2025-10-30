# YOUR CODE HERE
p, q = input().split()
positions = {'A': 0, 'B': 3, 'C': 4, 'D': 8, 'E': 9, 'F': 14, 'G': 23}
print(abs(positions[p] - positions[q]))