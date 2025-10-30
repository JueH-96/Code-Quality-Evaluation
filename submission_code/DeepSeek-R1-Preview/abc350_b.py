n, q = map(int, input().split())
t_list = list(map(int, input().split()))

# Initialize teeth: index 0 unused, 1..n are True (teeth present)
teeth = [True] * (n + 1)

for t in t_list:
    teeth[t] = not teeth[t]

# Sum the True values from 1 to n
print(sum(teeth[1:n+1]))