n = int(input())
k = n - 1

if k == 0:
    print(0)
else:
    digits = []
    while k > 0:
        digits.append(k % 5)
        k = k // 5
    digits = digits[::-1]
    res = ''.join(str(2 * d) for d in digits)
    print(res)