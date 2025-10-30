# YOUR CODE HERE
n = int(input())
a = list(map(int, input().split()))

operations = 0
while sum(x > 0 for x in a) > 1:
    a.sort(reverse=True)
    a[0] -= 1
    a[1] -= 1
    operations += 1

print(operations)