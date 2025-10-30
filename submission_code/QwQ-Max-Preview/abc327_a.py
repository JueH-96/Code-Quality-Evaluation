n = int(input())
s = input().strip()

found = False
for i in range(n - 1):
    if (s[i] == 'a' and s[i+1] == 'b') or (s[i] == 'b' and s[i+1] == 'a'):
        found = True
        break

print("Yes" if found else "No")