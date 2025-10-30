# YOUR CODE HERE

t = int(input().strip())

for _ in range(t):
    n = int(input().strip())
    a = list(map(int, input().strip().split()))
    
    # Find the digit that needs to be added
    for i in range(n):
        if a[i] < 9:
            a[i] += 1
            break
        else:
            a[i] = 0
    
    # Calculate the product of the remaining digits
    product = 1
    for num in a:
        product *= num
    
    print(product)