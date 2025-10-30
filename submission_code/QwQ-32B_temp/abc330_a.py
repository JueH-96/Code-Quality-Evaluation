n, l = map(int, input().split())
scores = list(map(int, input().split()))
count = 0
for score in scores:
    if score >= l:
        count += 1
print(count)