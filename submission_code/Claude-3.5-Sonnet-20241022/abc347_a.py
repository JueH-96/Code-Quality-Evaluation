N, K = map(int, input().split())
A = list(map(int, input().split()))

result = []
for num in A:
    if num % K == 0:
        result.append(num // K)

result.sort()
print(*result)