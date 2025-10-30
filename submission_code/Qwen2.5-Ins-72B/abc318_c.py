# YOUR CODE HERE
import sys

def min_cost(N, D, P, F):
    F.sort(reverse=True)
    total_cost = sum(F)
    min_cost = total_cost
    for i in range(0, N, D):
        total_cost -= sum(F[i:i+D])
        total_cost += P
        min_cost = min(min_cost, total_cost)
    return min_cost

input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
P = int(data[2])
F = list(map(int, data[3:]))

print(min_cost(N, D, P, F))