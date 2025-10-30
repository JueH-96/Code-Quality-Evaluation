n = int(input())
num = n - 1
digits = []
current = num
if current == 0:
    digits = [0]
else:
    while current > 0:
        digits.append(current % 5)
        current = current // 5
    digits = digits[::-1]
result = ''.join(str(d * 2) for d in digits)
print(result)