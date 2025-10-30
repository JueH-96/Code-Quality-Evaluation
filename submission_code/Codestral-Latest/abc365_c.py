import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
A = list(map(int, data[2:]))

A.sort()

total = 0
for i in range(N):
    total += A[i]
    if total > M:
        if i == 0:
            print("infinite")
        else:
            print(A[i-1])
        break
else:
    print("infinite")