n = int(input().strip())
s = input().strip()
total = 0
f = 0
for i in range(n):
    f = f * 10 + (i + 1) * (ord(s[i]) - ord('0'))
    total += f
print(total)