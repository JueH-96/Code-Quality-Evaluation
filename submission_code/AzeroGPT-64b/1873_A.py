t = int(input())

for _ in range(t):
    a = input().strip()
    if a in ("abc", "acb", "bac", "cba"):
        print("YES")
    else:
        print("NO")