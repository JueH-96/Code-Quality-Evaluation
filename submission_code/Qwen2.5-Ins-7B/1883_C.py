# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    a = list(map(int, input().split()))
    product = 1
    for num in a:
        product *= num
    operations = 0
    for i in range(k):
        if product % k == 0:
            break
        else:
            product *= (i + 1)
            operations += 1
    print(operations)