n = int(input())
a = n // 100
b = (n // 10) % 10
c = n % 10
first = b * 100 + c * 10 + a
second = c * 100 + a * 10 + b
print(first, second)