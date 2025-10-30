n, m = map(int, input().split())
family_has_son = [False] * (n + 1)  # 1-based indexing for families

for _ in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'M':
        if not family_has_son[a]:
            print("Yes")
            family_has_son[a] = True
        else:
            print("No")
    else:
        print("No")