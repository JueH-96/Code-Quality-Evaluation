t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    
    # If there's a 0, add 1 to it
    if 0 in a:
        # Find the first 0 and add 1 to it
        for i in range(n):
            if a[i] == 0:
                a[i] = 1
                break
    else:
        # Find the minimum element and add 1 to it
        min_val = min(a)
        for i in range(n):
            if a[i] == min_val:
                a[i] += 1
                break
    
    # Calculate the product
    product = 1
    for digit in a:
        product *= digit
    
    print(product)