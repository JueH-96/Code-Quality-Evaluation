n = int(input())
products = []
for _ in range(n):
    T, D = map(int, input().split())
    end = T + D
    products.append((end, T))

# Sort by end time, then by T
products.sort()

count = 0
last = 0

for end, T in products:
    earliest = last + 1
    if earliest >= T and earliest <= end:
        count += 1
        last = earliest

print(count)