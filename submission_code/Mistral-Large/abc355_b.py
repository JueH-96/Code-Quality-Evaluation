import sys

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

A = list(map(int, data[2:2+N]))
B = list(map(int, data[2+N:2+N+M]))

C = sorted(A + B)

consecutive_in_A = any(C[i] in A and C[i+1] in A for i in range(len(C)-1))

print("Yes" if consecutive_in_A else "No")