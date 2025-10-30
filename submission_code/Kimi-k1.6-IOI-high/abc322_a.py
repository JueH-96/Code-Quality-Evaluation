n = int(input())
s = input().strip()

for i in range(n - 2):
    if s[i] == 'A' and s[i+1] == 'B' and s[i+2] == 'C':
        print(i + 1)
        break
else:
    print(-1)