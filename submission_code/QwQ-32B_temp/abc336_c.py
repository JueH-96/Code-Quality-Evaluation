N = int(input())
n = N - 1
if n == 0:
    print(0)
else:
    digits = []
    current = n
    while current > 0:
        digits.append(current % 5)
        current = current // 5
    digits.reverse()
    result = ''.join(str(2 * d) for d in digits)
    print(result)