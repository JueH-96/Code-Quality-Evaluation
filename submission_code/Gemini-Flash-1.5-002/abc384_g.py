n = int(input())
a = list(map(int, input().split()))
b = list(map(int, input().split()))
k = int(input())

for _ in range(k):
    x, y = map(int, input().split())
    total_sum = 0
    for i in range(1, x + 1):
        for j in range(1, y + 1):
            total_sum += abs(a[i - 1] - b[j - 1])
    print(total_sum)