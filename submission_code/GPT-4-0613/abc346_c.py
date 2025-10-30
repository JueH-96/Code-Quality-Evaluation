N, K = map(int, input().split())
A = list(map(int, input().split()))

total = sum(range(1, K+1))
missing = sum(set(range(1, K+1)) - set(A))
print(total - missing)