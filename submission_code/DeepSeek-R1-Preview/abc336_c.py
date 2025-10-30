n = int(input())
m = n - 1

def to_base5(num):
    if num == 0:
        return '0'
    digits = []
    while num > 0:
        digits.append(num % 5)
        num = num // 5
    return ''.join(map(str, reversed(digits)))

base5_str = to_base5(m)
result = ''.join(str(int(c) * 2) for c in base5_str)
print(result)