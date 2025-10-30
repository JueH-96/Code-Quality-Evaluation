N = int(input())
A = list(map(int, input().split()))

i = 0
while True:
    if i >= len(A) - 1:
        break
    if abs(A[i] - A[i + 1]) == 1:
        i += 1
        continue
    if A[i] < A[i + 1]:
        for j in range(A[i] + 1, A[i + 1]):
            A.insert(i + 1, j)
            i += 1
    else:
        for j in range(A[i] - 1, A[i + 1], -1):
            A.insert(i + 1, j)
            i += 1
    i += 1

print(*A)