# YOUR CODE HERE
N, X, Y = map(int, input().split())
dishes = []
for i in range(N):
    a, b = map(int, input().split())
    dishes.append((a, b))

max_dishes = 0
import itertools
for perm in itertools.permutations(dishes):
    sweetness = 0
    saltiness = 0
    count = 0
    for a, b in perm:
        sweetness += a
        saltiness += b
        if sweetness > X or saltiness > Y:
            break
        count += 1
    max_dishes = max(max_dishes, count)

print(max_dishes)