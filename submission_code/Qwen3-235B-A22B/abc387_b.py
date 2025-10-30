x = int(input())
count = 0
for i in range(1, 10):
    if x % i == 0:
        j = x // i
        if 1 <= j <= 9:
            count += 1
total = 2025 - x * count
print(total)