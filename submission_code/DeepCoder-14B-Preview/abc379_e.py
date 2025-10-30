n = int(input())
s = input().strip()

total = 0
for i in range(n):
    m = n - (i + 1) + 1  # This simplifies to m = n - i
    power = (10 ** m) - 1
    sum_part = power // 9
    contribution = int(s[i]) * (i + 1) * sum_part
    total += contribution

print(total)