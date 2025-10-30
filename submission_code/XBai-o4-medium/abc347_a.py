n, k = map(int, input().split())
a = list(map(int, input().split()))
result = [str(num // k) for num in a if num % k == 0]
print(' '.join(result))