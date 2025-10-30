N = input().strip()

a = int(N[0])
b = int(N[1])
c = int(N[2])

first = b * 100 + c * 10 + a
second = c * 100 + a * 10 + b

print(first, second)