n = int(input())
s = input().strip()

count = 0

# Iterate over each possible starting seat i from 1 to N-2 inclusive
for i in range(1, n - 1):
    if s[i - 1] == '#' and s[i] == '.' and s[i + 1] == '#':
        count += 1

print(count)