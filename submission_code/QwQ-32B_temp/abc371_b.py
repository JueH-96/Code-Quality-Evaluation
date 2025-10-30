n, m = map(int, input().split())
first_male = [False] * (n + 1)  # Families are 1-based

for _ in range(m):
    a, b = input().split()
    a = int(a)
    if b == 'M':
        if not first_male[a]:
            print("Yes")
            first_male[a] = True
        else:
            print("No")
    else:
        print("No")