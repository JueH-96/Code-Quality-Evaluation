n = int(input())
s = [input().strip() for _ in range(n)]

for i in range(n-2):
    if s[i] == 'sweet' and s[i+1] == 'sweet':
        print("No")
        exit()

print("Yes")