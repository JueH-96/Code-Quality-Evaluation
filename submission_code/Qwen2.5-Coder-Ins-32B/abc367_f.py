# YOUR CODE HERE
import sys
from collections import Counter

input = sys.stdin.read().split()
N = int(input[0])
Q = int(input[1])
A = list(map(int, input[2:2+N]))
B = list(map(int, input[2+N:2+2*N]))
queries = []

index = 2 + 2 * N
for _ in range(Q):
    l_i = int(input[index]) - 1
    r_i = int(input[index + 1]) - 1
    L_i = int(input[index + 2]) - 1
    R_i = int(input[index + 3]) - 1
    queries.append((l_i, r_i, L_i, R_i))
    index += 4

def can_rearrange(subseq_A, subseq_B):
    return Counter(subseq_A) == Counter(subseq_B)

results = []
for l_i, r_i, L_i, R_i in queries:
    if can_rearrange(A[l_i:r_i+1], B[L_i:R_i+1]):
        results.append("Yes")
    else:
        results.append("No")

print("
".join(results))