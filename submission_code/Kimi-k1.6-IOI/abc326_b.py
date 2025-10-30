n = int(input())
for num in range(n, 1000):
    a = num // 100
    b = (num // 10) % 10
    c = num % 10
    if a * b == c:
        print(num)
        break