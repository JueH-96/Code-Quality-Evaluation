S = input().strip()

allowed = {'A', 'B', 'C'}
for c in S:
    if c not in allowed:
        print("No")
        exit()
    if c == 'B':
        allowed = {'B', 'C'}
    elif c == 'C':
        allowed = {'C'}
print("Yes")