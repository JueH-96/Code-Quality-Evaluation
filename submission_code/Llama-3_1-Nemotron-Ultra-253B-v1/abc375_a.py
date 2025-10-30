n = int(input())
s = input().strip()
count = 0
for j in range(n - 2):
    if s[j] == '#' and s[j+1] == '.' and s[j+2] == '#':
        count += 1
print(count)