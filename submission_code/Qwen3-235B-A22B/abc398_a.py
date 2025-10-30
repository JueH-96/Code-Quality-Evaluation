n = int(input())
s = ['-'] * n

if n % 2 == 0:
    mid = n // 2
    s[mid - 1] = '='
    s[mid] = '='
else:
    mid = n // 2
    s[mid] = '='

print(''.join(s))