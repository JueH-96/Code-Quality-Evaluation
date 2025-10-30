# YOUR CODE HERE
N, X, Y = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

dishes = []
for i in range(N):
    dishes.append((A[i], B[i]))

import itertools

min_dishes = float('inf')
for perm in itertools.permutations(dishes):
    sweetness_sum = 0
    saltiness_sum = 0
    count = 0
    for i in range(len(perm)):
        sweetness_sum += perm[i][0]
        saltiness_sum += perm[i][1]
        count += 1
        if sweetness_sum > X or saltiness_sum > Y:
            min_dishes = min(min_dishes, count)
            break
    if sweetness_sum <= X and saltiness_sum <=Y:
        min_dishes = min(min_dishes,count)

print(min_dishes)