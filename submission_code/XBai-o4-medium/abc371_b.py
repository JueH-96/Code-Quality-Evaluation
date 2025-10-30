n, m = map(int, input().split())
has_eldest = [False] * (n + 1)  # 1-based indexing

for _ in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'M':
        if not has_eldest[a]:
            print("Yes")
            has_eldest[a] = True
        else:
            print("No")
    else:
        print("No")