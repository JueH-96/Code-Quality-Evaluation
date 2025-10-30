n = int(input())
products = []
for _ in range(n):
    T, D = map(int, input().split())
    end = T + D
    products.append((T, end))

# Sort products based on their end times
products.sort(key=lambda x: x[1])

last_print = -1
count = 0

for T, end in products:
    earliest = max(T, last_print + 1)
    if earliest < end:
        count += 1
        last_print = earliest

print(count)