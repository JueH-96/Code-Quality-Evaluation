# YOUR CODE HERE
distances = {
    ('A', 'B'): 3,
    ('B', 'C'): 1,
    ('C', 'D'): 4,
    ('D', 'E'): 1,
    ('E', 'F'): 5,
    ('F', 'G'): 9
}

points = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def find_distance(p, q):
    if points.index(p) > points.index(q):
        p, q = q, p
    distance = 0
    current = p
    while current != q:
        next_point = points[points.index(current) + 1]
        distance += distances[(current, next_point)]
        current = next_point
    return distance

import sys
p, q = sys.stdin.read().strip().split()
print(find_distance(p, q))