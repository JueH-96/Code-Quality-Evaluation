n, L, R = map(int, input().split())
a = list(map(int, input().split()))
result = [max(L, min(R, num)) for num in a]
print(' '.join(map(str, result)))