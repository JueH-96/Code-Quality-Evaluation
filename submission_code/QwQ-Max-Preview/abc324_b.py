# Read input
n = int(input())

# Remove all factors of 2
x = 0
while n % 2 == 0:
    n = n // 2
    x += 1

# Remove all factors of 3
y = 0
while n % 3 == 0:
    n = n // 3
    y += 1

# Check if remaining number is 1
if n == 1:
    print("Yes")
else:
    print("No")