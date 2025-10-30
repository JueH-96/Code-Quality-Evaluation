n, x = map(int, input().split())
a = list(map(int, input().split()))

min_score = None

for s in range(101):
    b = a + [s]
    b.sort()
    current_sum = sum(b[1 : n-1])
    if current_sum >= x:
        if min_score is None or s < min_score:
            min_score = s

print(min_score if min_score is not None else -1)