n, l = map(int, input().split())
scores = list(map(int, input().split()))
count = sum(1 for score in scores if score >= l)
print(count)