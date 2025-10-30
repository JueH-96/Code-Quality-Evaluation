n = int(input())
s = input().strip()
t = input().strip()

for a, b in zip(s, t):
    if a == b:
        continue
    if (a in {'1', 'l'}) and (b in {'1', 'l'}):
        continue
    if (a in {'0', 'o'}) and (b in {'0', 'o'}):
        continue
    print("No")
    exit()
print("Yes")