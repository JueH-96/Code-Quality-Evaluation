n, m = map(int, input().split())
has_first_son = [False] * (n + 1)

for _ in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'M':
        if not has_first_son[a]:
            print("Yes")
            has_first_son[a] = True
        else:
            print("No")
    else:
        print("No")