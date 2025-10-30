N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

i = N - 1
j = M - 1
max_sum = -1

while i >= 0 and j >= 0:
    if abs(A[i] - B[j]) <= D:
        max_sum = max(max_sum, A[i] + B[j])
        i -= 1
        j -= 1
    elif A[i] > B[j]:
        i -= 1
    else:
        j -= 1

print(max_sum)