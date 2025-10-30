n = int(input())
s = [input().strip() for _ in range(n)]

for i in range(len(s)-1):
    if s[i] == 's' and s[i+1] == 's':
        print("No")
        exit()

print("Yes")