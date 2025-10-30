n, m = map(int, input().split())
family_has_male = [False] * (n + 1)

for _ in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'F':
        print("No")
    else:
        if not family_has_male[a]:
            print("Yes")
            family_has_male[a] = True
        else:
            print("No")