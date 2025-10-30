def max_product(digits):
    # Count the number of 0s
    zero_count = digits.count(0)
    
    # If there are multiple 0s, the product will still be 0 even after adding 1 to one of them
    if zero_count > 1:
        return 0
    
    # If there's a single 0, add 1 to it
    if zero_count == 1:
        # Calculate the product after replacing the 0 with a 1
        product = 1
        for d in digits:
            if d != 0:
                product *= d
        return product
    
    # If there are no 0s, add 1 to the smallest digit
    min_digit = min(digits)
    min_idx = digits.index(min_digit)
    
    # Calculate the product with the incremented digit
    product = 1
    for i, d in enumerate(digits):
        if i == min_idx:
            product *= (d + 1)
        else:
            product *= d
    
    return product

t = int(input())
for _ in range(t):
    n = int(input())
    digits = list(map(int, input().split()))
    print(max_product(digits))