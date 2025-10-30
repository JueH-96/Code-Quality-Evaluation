n = int(input())
s = input().strip()
t = input().strip()

for i in range(n):
    sc = s[i]
    tc = t[i]
    if sc == tc:
        continue
    if (sc == '1' and tc == 'l') or (sc == 'l' and tc == '1'):
        continue
    if (sc == '0' and tc == 'o') or (sc == 'o' and tc == '0'):
        continue
    print("No")
    exit()
print("Yes")