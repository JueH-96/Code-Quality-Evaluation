n, q = map(int, input().split())
treatments = list(map(int, input().split()))
teeth = [True] * n
for t in treatments:
    teeth[t - 1] = not teeth[t - 1]
count = sum(teeth)
print(count)