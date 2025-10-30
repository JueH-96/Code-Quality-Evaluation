n, m = map(int, input().split())
x = list(map(int, input().split()))
a = list(map(int, input().split()))

stones = [0] * (n + 1)
for i in range(m):
    stones[x[i]] = a[i]

total_stones = sum(stones)
if total_stones < n:
    print(-1)
else:
    operations = 0
    current_stones = 0
    for i in range(1, n + 1):
        current_stones += stones[i]
        if current_stones < i:
            print(-1)
            exit()
        elif current_stones > i:
            operations += current_stones - i
    print(operations)