n = int(input())
m = n - 1

if m == 0:
    print(0)
else:
    digits = []
    while m > 0:
        digits.append(m % 5)
        m = m // 5
    digits = digits[::-1]  # Reverse to get the correct order
    res = 0
    for d in digits:
        res = res * 10 + (d * 2)
    print(res)