n = int(input())
s = input().strip()

count = 0
for i in range(n - 2):
    if s[i] == '#' and s[i+1] == '.' and s[i+2] == '#':
        count += 1
print(count)