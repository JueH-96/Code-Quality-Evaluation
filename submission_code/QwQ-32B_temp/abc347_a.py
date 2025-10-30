n, k = map(int, input().split())
a = list(map(int, input().split()))
result = []
for num in a:
    if num % k == 0:
        result.append(num // k)
print(' '.join(map(str, result)))