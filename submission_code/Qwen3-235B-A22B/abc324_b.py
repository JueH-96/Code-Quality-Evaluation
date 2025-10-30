n = int(input())
current = n

# Remove all factors of 2
x = 0
while current % 2 == 0:
    current //= 2
    x += 1

# Remove all factors of 3
y = 0
while current % 3 == 0:
    current //= 3
    y += 1

# Check if remaining number is 1
if current == 1:
    print("Yes")
else:
    print("No")