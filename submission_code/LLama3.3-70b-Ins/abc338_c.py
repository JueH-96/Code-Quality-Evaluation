import sys

def max_servings(N, Q, A, B):
    max_servings = 0
    for x in range(Q[0] // A[0] + 1):
        remaining_Q = [q - x * a for q, a in zip(Q, A)]
        servings_B = min([r // b for r, b in zip(remaining_Q, B)])
        max_servings = max(max_servings, x + servings_B)
    return max_servings

N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

print(max_servings(N, Q, A, B))