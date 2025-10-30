# YOUR CODE HERE
def min_dishes_to_eat(N, X, Y, A, B):
    dishes = list(zip(A, B))
    dishes.sort(key=lambda x: (x[0], x[1]))

    total_sweetness = 0
    total_saltiness = 0
    count = 0

    for sweetness, saltiness in dishes:
        total_sweetness += sweetness
        total_saltiness += saltiness
        count += 1
        if total_sweetness > X or total_saltiness > Y:
            break

    return count

import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
X = int(data[1])
Y = int(data[2])
A = list(map(int, data[3:3+N]))
B = list(map(int, data[3+N:]))

print(min_dishes_to_eat(N, X, Y, A, B))