n = int(input())
s = ['-'] * n
if n % 2 == 1:
    mid = n // 2
    s[mid] = '='
else:
    mid_left = (n // 2) - 1
    s[mid_left] = '='
    s[mid_left + 1] = '='
print(''.join(s))