# YOUR CODE HERE
from itertools import permutations

def max_dishes(N, X, Y, dishes):
    max_eaten = 0
    for perm in permutations(dishes):
        total_sweetness = 0
        total_saltiness = 0
        count = 0
        for sweetness, saltiness in perm:
            if total_sweetness + sweetness > X or total_saltiness + saltiness > Y:
                break
            total_sweetness += sweetness
            total_saltiness += saltiness
            count += 1
        max_eaten = max(max_eaten, count)
    return max_eaten

import sys
input = sys.stdin.read().split()
N = int(input[0])
X = int(input[1])
Y = int(input[2])
dishes = [(int(input[2*i+3]), int(input[2*i+4])) for i in range(N)]

print(max_dishes(N, X, Y, dishes))