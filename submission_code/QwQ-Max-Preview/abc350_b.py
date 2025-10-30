n, q = map(int, input().split())
treatments = list(map(int, input().split()))

# Initialize teeth: hole 1 to N start with True (has tooth)
teeth = [False] + [True] * n

for t in treatments:
    teeth[t] = not teeth[t]

print(sum(teeth[1:n+1]))