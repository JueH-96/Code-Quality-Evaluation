n = int(input())
s = [input().strip() for _ in range(n)]
possible = True
for i in range(n-1):
    if s[i] == 'sweet' and s[i+1] == 'sweet':
        if (i+1) != (n-1):
            possible = False
            break
print("Yes" if possible else "No")