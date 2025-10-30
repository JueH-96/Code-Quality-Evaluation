n = int(input())
p = list(map(int, input().split()))

total_cost = 0

for i in range(1, n):
    count = 0
    for j in range(i):
        if p[j] > i:
            count += 1
    total_cost += i * count

print(total_cost)