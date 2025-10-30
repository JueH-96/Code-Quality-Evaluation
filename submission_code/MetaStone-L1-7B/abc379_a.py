n = int(input())
a = n // 100
b = (n // 10) % 10
c = n % 10

first = int(f"{b}{c}{a}")
second = int(f"{c}{a}{b}")

print(first, second)