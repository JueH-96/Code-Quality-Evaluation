n, m = map(int, input().split())
has_male = [False] * (n + 1)  # 1-based indexing for families

for _ in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'M' and not has_male[a]:
        print("Yes")
        has_male[a] = True
    else:
        print("No")