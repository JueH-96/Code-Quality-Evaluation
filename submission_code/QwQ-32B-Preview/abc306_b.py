A = list(map(int, input().split()))
result = 0
for i in range(64):
    result += A[i] * (1 << i)
print(result)