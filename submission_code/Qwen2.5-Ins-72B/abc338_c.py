# YOUR CODE HERE
def max_servings(N, Q, A, B):
    from itertools import product
    
    max_s = 0
    for a in range(N + 1):
        for b in range(N + 1):
            if all(Q[i] >= A[i] * a + B[i] * b for i in range(N)):
                max_s = max(max_s, a + b)
    return max_s

N = int(input())
Q = list(map(int, input().split()))
A = list(map(int, input().split()))
B = list(map(int, input().split()))

print(max_servings(N, Q, A, B))