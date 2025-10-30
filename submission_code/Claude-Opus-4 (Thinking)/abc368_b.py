n = int(input())
a = list(map(int, input().split()))

operations = 0

while sum(1 for x in a if x > 0) > 1:
    a.sort(reverse=True)
    a[0] -= 1
    a[1] -= 1
    operations += 1

print(operations)