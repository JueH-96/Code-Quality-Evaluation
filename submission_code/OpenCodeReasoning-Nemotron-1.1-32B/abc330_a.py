n, l = map(int, input().split())
scores = list(map(int, input().split()))
count = sum(1 for s in scores if s >= l)
print(count)