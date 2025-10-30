n = int(input())
a = list(map(int, input().split()))
count = 0

while True:
    # Count the number of positive elements
    positive_count = sum(1 for x in a if x > 0)
    if positive_count <= 1:
        break
    # Perform the operation
    a = sorted(a, reverse=True)
    a[0] -= 1
    a[1] -= 1
    count += 1

print(count)