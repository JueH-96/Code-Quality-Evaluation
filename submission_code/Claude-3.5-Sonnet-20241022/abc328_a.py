N, X = map(int, input().split())
S = list(map(int, input().split()))

total = 0
for score in S:
    if score <= X:
        total += score

print(total)