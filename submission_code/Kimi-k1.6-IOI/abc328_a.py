n, x = map(int, input().split())
scores = list(map(int, input().split()))
total = sum(score for score in scores if score <= x)
print(total)