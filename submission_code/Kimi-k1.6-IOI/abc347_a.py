n, k = map(int, input().split())
a = list(map(int, input().split()))
result = [str(x // k) for x in a if x % k == 0]
print(' '.join(result))