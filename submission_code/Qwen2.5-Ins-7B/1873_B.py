# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    product = 1
    for digit in a:
        product *= digit
    max_digit = max(a)
    if max_digit != 9:
        new_product = product * (max_digit + 1) // max_digit
        product = max(product, new_product)
    else:
        for i in range(n):
            if a[i] != 9:
                new_product = product // a[i] * (a[i] + 1)
                product = max(product, new_product)
                break
    print(product)