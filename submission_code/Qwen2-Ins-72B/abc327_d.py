N, M = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

same = set()
for i in range(M):
    if A[i] == B[i]:
        same.add(A[i])

if len(same) > 0:
    if len(same) == N:
        print("No")
    else:
        print("Yes")
else:
    print("Yes")