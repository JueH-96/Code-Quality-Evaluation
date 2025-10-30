def is_square(n):
    if n == 0:
        return True
    i = 1
    while i * i <= n:
        if i * i == n:
            return True
        i += 1
    return False

N = int(input())
A = list(map(int, input().split()))

count = 0
for i in range(N):
    for j in range(i+1, N):
        if is_square(A[i] * A[j]):
            count += 1

print(count)