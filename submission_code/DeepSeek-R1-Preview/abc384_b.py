n, r = map(int, input().split())
contests = []
for _ in range(n):
    d, a = map(int, input().split())
    contests.append((d, a))

current_rating = r

for d, a in contests:
    if d == 1:
        if 1600 <= current_rating <= 2799:
            current_rating += a
    else:
        if 1200 <= current_rating <= 2399:
            current_rating += a

print(current_rating)