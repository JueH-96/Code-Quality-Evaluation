n, m = map(int, input().split())
A = list(map(int, input().split()))
sum_b = [0] * m
for _ in range(n):
    X = list(map(int, input().split()))
    for j in range(m):
        sum_b[j] += X[j]
if all(sum_b[j] >= A[j] for j in range(m)):
    print("Yes")
else:
    print("No")