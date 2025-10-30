n = int(input())

# Divide n by 2 until it's no longer divisible by 2
while n % 2 == 0:
    n = n // 2

# Divide n by 3 until it's no longer divisible by 3
while n % 3 == 0:
    n = n // 3

# If n is reduced to 1, it only had 2 and 3 as prime factors
if n == 1:
    print("Yes")
else:
    print("No")