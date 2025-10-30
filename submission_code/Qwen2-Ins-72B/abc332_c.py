n,m = map(int,input().split())
s = input()
c = 0
p = 0
for i in range(n):
    if s[i] == '2':
        c += 1
        p = 0
    elif s[i] == '1':
        p += 1
print(max(0,max(c,p)-m))