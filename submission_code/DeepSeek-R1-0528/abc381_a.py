n = int(input().strip())
s = input().strip()

if n % 2 != 1:
    print("No")
    exit()

mid = (n - 1) // 2
if s[mid] != '/':
    print("No")
    exit()

for i in range(mid):
    if s[i] != '1':
        print("No")
        exit()

for i in range(mid + 1, n):
    if s[i] != '2':
        print("No")
        exit()

print("Yes")