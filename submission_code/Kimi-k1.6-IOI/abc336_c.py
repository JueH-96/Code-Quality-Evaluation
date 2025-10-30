n = int(input())
m = n - 1
digits = []
if m == 0:
    digits = [0]
else:
    while m > 0:
        digits.append(m % 5)
        m = m // 5
    digits = digits[::-1]  # Reverse to get the most significant digit first
result = ''.join(str(2 * d) for d in digits)
print(result)