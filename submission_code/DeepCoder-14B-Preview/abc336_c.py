def to_base5(n):
    if n == 0:
        return '0'
    digits = []
    while n > 0:
        digits.append(n % 5)
        n = n // 5
    digits = digits[::-1]
    return ''.join(str(d) for d in digits)

n = int(input())
if n == 0:
    print(0)
else:
    m = n - 1
    base5 = to_base5(m)
    even_digits = ['0', '2', '4', '6', '8']
    res = []
    for c in base5:
        d = int(c)
        res.append(even_digits[d])
    print(''.join(res))