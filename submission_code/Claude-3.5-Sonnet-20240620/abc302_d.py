# YOUR CODE HERE
N, M, D = map(int, input().split())
A = list(map(int, input().split()))
B = list(map(int, input().split()))

A.sort(reverse=True)
B.sort(reverse=True)

i, j = 0, 0
max_sum = -1

while i < N and j < M:
    if abs(A[i] - B[j]) <= D:
        max_sum = max(max_sum, A[i] + B[j])
        break
    elif A[i] > B[j]:
        i += 1
    else:
        j += 1

print(max_sum)