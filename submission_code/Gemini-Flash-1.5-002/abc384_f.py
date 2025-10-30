def f(x):
    while x % 2 == 0 and x > 0:
        x //= 2
    return x

N = int(input())
A = list(map(int, input().split()))

total_sum = 0
for i in range(N):
    for j in range(i, N):
        total_sum += f(A[i] + A[j])

print(total_sum)