n, l = map(int, input().split())
scores = list(map(int, input().split()))
count = sum(1 for a in scores if a >= l)
print(count)