N = int(input())
A = list(map(int, input().split()))

A.sort()
max_num = A[-1]

for i in range(N-1,-1,-1):
    if A[i] != max_num:
        print(A[i])
        break