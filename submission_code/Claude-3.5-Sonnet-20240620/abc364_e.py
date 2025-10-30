# YOUR CODE HERE
import sys
from itertools import permutations

def max_dishes_eaten(N, X, Y, dishes):
    max_eaten = 0
    for perm in permutations(dishes):
        eaten = 0
        total_sweet = 0
        total_salty = 0
        for sweet, salty in perm:
            if total_sweet + sweet > X or total_salty + salty > Y:
                break
            total_sweet += sweet
            total_salty += salty
            eaten += 1
        max_eaten = max(max_eaten, eaten)
    return max_eaten

# Read input
N, X, Y = map(int, input().split())
dishes = [tuple(map(int, input().split())) for _ in range(N)]

# Solve and print result
print(max_dishes_eaten(N, X, Y, dishes))