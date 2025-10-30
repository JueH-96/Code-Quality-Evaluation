n = int(input())
x = 0
while n % 2 == 0:
    x += 1
    n = n // 2
y = 0
while n % 3 == 0:
    y += 1
    n = n // 3
print("Yes" if n == 1 else "No")