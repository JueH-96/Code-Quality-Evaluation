n, m = map(int, input().split())
sum_x = 0
current = 1
for _ in range(m + 1):
    sum_x += current
    if sum_x > 10**9:
        print("inf")
        exit()
    current *= n
print(sum_x)