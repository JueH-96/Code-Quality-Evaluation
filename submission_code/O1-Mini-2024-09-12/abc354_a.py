H = int(input())

n = 0
current = 0

while current <= H:
    current += 2**n
    n += 1

print(n)