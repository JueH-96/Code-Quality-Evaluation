# YOUR CODE HERE
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
Q = int(data[1])

A = list(map(int, data[2:2+N]))
B = list(map(int, data[2+N:2+N+N]))

queries = []
index = 2*N+N
for _ in range(Q):
    l_i = int(data[index])
    r_i = int(data[index+1])
    L_i = int(data[index+2])
    R_i = int(data[index+3])
    queries.append((l_i, r_i, L_i, R_i))
    index += 4

for l_i, r_i, L_i, R_i in queries:
    sub_A = A[l_i-1:r_i]
    sub_B = B[L_i-1:R_i]
    if sorted(sub_A) == sorted(sub_B):
        print("Yes")
    else:
        print("No")