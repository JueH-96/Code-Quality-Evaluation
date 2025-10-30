X = int(input())
total = 45 * 45
count = 0
for i in range(1, 10):
    for j in range(1, 10):
        if i * j == X:
            count += 1
ans = total - count * X
print(ans)