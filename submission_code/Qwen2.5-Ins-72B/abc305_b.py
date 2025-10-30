# YOUR CODE HERE
points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
distances = [3, 1, 4, 1, 5, 9]

p, q = input().split()

p_index = points.index(p)
q_index = points.index(q)

if p_index > q_index:
    p_index, q_index = q_index, p_index

distance = sum(distances[p_index:q_index])
print(distance)