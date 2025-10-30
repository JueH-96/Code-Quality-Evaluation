import sys
from itertools import permutations

# Read input
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])

dishes = []
for i in range(N):
    A = int(data[3 + 2 * i])
    B = int(data[4 + 2 * i])
    dishes.append((A, B))

# Function to find the maximum number of dishes Snuke can eat
def max_dishes(dishes, X, Y):
    max_eaten = 0
    for perm in permutations(dishes):
        total_sweetness = 0
        total_saltiness = 0
        eaten = 0
        for A, B in perm:
            total_sweetness += A
            total_saltiness += B
            if total_sweetness > X or total_saltiness > Y:
                break
            eaten += 1
        max_eaten = max(max_eaten, eaten)
    return max_eaten

# Output the result
print(max_dishes(dishes, X, Y))