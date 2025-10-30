H = int(input())
target = H + 1
n = 0
current = 1
while current <= target:
    current *= 2
    n += 1
print(n)