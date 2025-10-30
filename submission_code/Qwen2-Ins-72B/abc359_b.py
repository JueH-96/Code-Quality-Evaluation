N = int(input())
A = list(map(int, input().split()))

cnt = 0
for i in range(N):
    index1 = A.index(i+1)
    A[index1] = 0
    index2 = A.index(i+1)
    if index1 < N and index2 > N-1 or index1 > N-1 and index2 < N:
        cnt += 1
print(cnt)