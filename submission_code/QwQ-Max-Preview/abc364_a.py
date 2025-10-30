n = int(input())
s = [input().strip() for _ in range(n)]

for i in range(n - 1):
    if s[i] == 'sweet' and s[i + 1] == 'sweet':
        if i != n - 2:
            print("No")
            exit()

print("Yes")