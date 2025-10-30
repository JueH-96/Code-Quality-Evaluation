n = int(input())
if n == 0:
    print(0)
else:
    if n <= 999:
        k = 0
    elif n <= 9999:
        k = 1
    elif n <= 99999:
        k = 2
    elif n <= 999999:
        k = 3
    elif n <= 9999999:
        k = 4
    elif n <= 99999999:
        k = 5
    else:
        k = 6
    s = str(n)
    if k == 0:
        print(s)
    else:
        new_s = s[:-k] + '0' * k
        print(new_s)