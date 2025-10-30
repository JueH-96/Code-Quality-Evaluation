n = int(input())
s = input().strip()
t = input().strip()

for i in range(n):
    a = s[i]
    b = t[i]
    if a == b:
        continue
    if (a == '1' and b == 'l') or (a == 'l' and b == '1'):
        continue
    if (a == '0' and b == 'o') or (a == 'o' and b == '0'):
        continue
    print("No")
    exit()

print("Yes")