n = int(input())
if n == 1:
    print(0)
else:
    m = n - 1
    digits = []
    if m == 0:
        digits = [0]
    else:
        while m > 0:
            digits.append(m % 5)
            m = m // 5
        digits = digits[::-1]
    result = ''.join(str(d * 2) for d in digits)
    print(result)