X = int(input())
total_sum = 0
for i in range(1, 10):
    for j in range(1, 10):
        if i * j != X:
            total_sum += i * j
print(total_sum)