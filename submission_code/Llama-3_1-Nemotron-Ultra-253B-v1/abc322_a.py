n = int(input())
s = input().strip()

for i in range(len(s) - 2):
    if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
        print(i + 1)
        exit()

print(-1)