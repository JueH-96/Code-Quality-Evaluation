n = int(input())
a = n // 100
b = (n // 10) % 10
c = n % 10
print(f"{b * 100 + c * 10 + a} {c * 100 + a * 10 + b}")