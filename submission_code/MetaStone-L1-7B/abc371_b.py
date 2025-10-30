n, m = map(int, input().split())
counts = [0] * (n + 1)
for _ in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'M':
        if counts[a] == 0:
            print("Yes")
            counts[a] += 1
        else:
            print("No")
    else:
        print("No")