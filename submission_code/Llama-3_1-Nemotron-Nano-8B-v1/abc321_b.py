n, x = map(int, input().split())
a = list(map(int, input().split()))

min_y = 101

for y in range(0, 101):
    b = a + [y]
    b_sorted = sorted(b)
    current_sum = sum(b_sorted[1:-1])
    if current_sum >= x and y < min_y:
        min_y = y

print(min_y if min_y <= 100 else -1)