n, m = map(int, input().split())
families = [False] * (n + 1)  # 1-based indexing

for _ in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'M':
        if not families[a]:
            print("Yes")
            families[a] = True
        else:
            print("No")
    else:
        print("No")