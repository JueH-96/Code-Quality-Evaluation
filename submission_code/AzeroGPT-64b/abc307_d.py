n = int(input())
s = input()
l, r = s.count(')'), s.count('(')
if l < r:
    print(s[l:])
elif l > r:
    print(s[:-r])
else:
    if l == 0:
        print("")
    else:
        print(s[l:-l])