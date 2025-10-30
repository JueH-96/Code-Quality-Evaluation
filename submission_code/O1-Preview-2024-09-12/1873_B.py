# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    max_product = 0
    for i in range(n):
        a_copy = a[:]
        a_copy[i] += 1
        prod = 1
        for num in a_copy:
            prod *= num
        max_product = max(max_product, prod)
    print(max_product)