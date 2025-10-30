n, m = map(int, input().split())
taro = [False] * (n + 1)  # 1-based indexing for families

for _ in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'M' and not taro[a]:
        print("Yes")
        taro[a] = True
    else:
        print("No")