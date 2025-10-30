# YOUR CODE HERE
import sys

def max_currency(N, A, S, T):
    for i in range(N - 2, -1, -1):
        A[i + 1] += (A[i] // S[i]) * T[i]
        A[i] %= S[i]
    return A[N - 1]

N = int(input())
A = list(map(int, input().split()))
S = []
T = []

for _ in range(N - 1):
    s, t = map(int, input().split())
    S.append(s)
    T.append(t)

print(max_currency(N, A, S, T))