t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    max_product = 0
    for i in range(n):
        temp = a[:]  # Create a copy of the array
        temp[i] += 1  # Increment the current digit
        product = 1
        for num in temp:
            product *= num
        if product > max_product:
            max_product = product
    print(max_product)