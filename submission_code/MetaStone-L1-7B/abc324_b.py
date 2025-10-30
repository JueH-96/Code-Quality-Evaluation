n = int(input())
x = 0
y = 0

# Divide by 2
while n % 2 == 0:
    n = n // 2
    x += 1

# Divide by 3
while n % 3 == 0:
    n = n // 3
    y += 1

print("Yes" if n == 1 else "No")