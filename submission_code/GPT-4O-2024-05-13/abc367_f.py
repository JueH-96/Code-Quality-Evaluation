# YOUR CODE HERE
import sys
input = sys.stdin.read
from collections import Counter

data = input().split()
N = int(data[0])
Q = int(data[1])
A = list(map(int, data[2:N+2]))
B = list(map(int, data[N+2:2*N+2]))

queries = []
for i in range(Q):
    l_i = int(data[2*N+2 + 4*i])
    r_i = int(data[2*N+3 + 4*i])
    L_i = int(data[2*N+4 + 4*i])
    R_i = int(data[2*N+5 + 4*i])
    queries.append((l_i, r_i, L_i, R_i))

results = []
for l_i, r_i, L_i, R_i in queries:
    sub_A = A[l_i-1:r_i]
    sub_B = B[L_i-1:R_i]
    if Counter(sub_A) == Counter(sub_B):
        results.append("Yes")
    else:
        results.append("No")

print("
".join(results))