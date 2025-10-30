N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort()
B.sort()

ans = -1
i = N-1
j = M-1

while i >= 0 and j >= 0:
    if abs(A[i] - B[j]) <= D:
        ans = max(ans, A[i] + B[j])
        i -= 1
    else:
        if A[i] > B[j]:
            i -= 1
        else:
            j -= 1

print(ans)