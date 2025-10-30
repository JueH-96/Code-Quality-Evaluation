n = int(input())
s = input().strip()

for i in range(len(s) - 2):
    if s[i:i+3] == 'ABC':
        print(i + 1)
        exit()

print(-1)