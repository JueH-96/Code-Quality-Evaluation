import sys

def min_cost_to_sort(N, P):
    cost = 0
    for i in range(N-1):
        if P[i] != i+1:
            for j in range(i, N):
                if P[j] == i+1:
                    cost += j-i
                    P[i:j+1] = reversed(P[i:j+1])
                    break
    return cost

N = int(input())
P = [int(x) for x in input().split()]
print(min_cost_to_sort(N, P))