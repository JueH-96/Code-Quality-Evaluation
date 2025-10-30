def f(x, y):
    return (x + y) % (10 ** 8)

N = int(input())
A = list(map(int, input().split()))

total = 0
for i in range(N):
    for j in range(i + 1, N):
        total += f(A[i], A[j])

print(total)