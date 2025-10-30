n, q = map(int, input().split())
t_list = list(map(int, input().split()))

# Initialize teeth: index 0 unused, 1..n are True (present)
teeth = [False] + [True] * n

for t in t_list:
    teeth[t] = not teeth[t]

print(sum(teeth[1:n+1]))