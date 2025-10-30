n = int(input())

# Find the maximum x such that x^3 <= n
max_x = int(n ** (1/3))
while (max_x + 1) ** 3 <= n:
    max_x += 1
while max_x ** 3 > n:
    max_x -= 1

# Check palindromic cubes from largest to smallest
for x in range(max_x, 0, -1):
    k = x ** 3
    s = str(k)
    if s == s[::-1]:
        print(k)
        exit()

# Fallback to 1 (shouldn't be reached as x=1 is always valid)
print(1)