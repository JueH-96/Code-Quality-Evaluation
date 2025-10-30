H = int(input())
target = H + 1
current = 1
n = 0
while current <= target:
    current *= 2
    n += 1
print(n)