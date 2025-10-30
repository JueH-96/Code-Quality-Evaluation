N = int(input())
s = str(N)
d = len(s)
if d <= 3:
    print(N)
else:
    truncate = d - 3
    prefix = s[:-truncate] if truncate != 0 else s
    print(prefix + '0' * truncate)