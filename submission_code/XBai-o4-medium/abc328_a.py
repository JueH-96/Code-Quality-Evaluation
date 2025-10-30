n, x = map(int, input().split())
scores = list(map(int, input().split()))
total = sum(s for s in scores if s <= x)
print(total)