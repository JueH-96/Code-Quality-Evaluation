# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    
    # Find the minimum element
    min_val = min(a)
    
    # Find the first occurrence of the minimum and add 1 to it
    for i in range(n):
        if a[i] == min_val:
            a[i] += 1
            break
    
    # Calculate the product
    product = 1
    for x in a:
        product *= x
    
    print(product)