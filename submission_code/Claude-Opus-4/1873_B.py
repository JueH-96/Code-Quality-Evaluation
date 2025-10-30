# YOUR CODE HERE
t = int(input())
for _ in range(t):
    n = int(input())
    a = list(map(int, input().split()))
    
    max_product = 0
    
    # Try adding 1 to each digit
    for i in range(n):
        if a[i] < 9:  # Can only add 1 if digit is less than 9
            # Create a copy of the array
            temp = a[:]
            temp[i] += 1
            
            # Calculate product
            product = 1
            for digit in temp:
                product *= digit
            
            max_product = max(max_product, product)
    
    print(max_product)