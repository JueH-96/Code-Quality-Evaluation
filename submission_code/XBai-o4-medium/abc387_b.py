X = int(input())
count = 0
for i in range(1, 10):
    if X % i == 0:
        j = X // i
        if 1 <= j <= 9:
            count += 1
print(2025 - count * X)