n, m = map(int, input().split())
assigned = [False] * n
for _ in range(m):
    a, b = input().split()
    a = int(a) - 1  # Convert to 0-based index
    if b == 'M' and not assigned[a]:
        print("Yes")
        assigned[a] = True
    else:
        print("No")