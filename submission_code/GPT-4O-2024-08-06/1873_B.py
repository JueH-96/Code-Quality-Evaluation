# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    t = int(data[index])
    index += 1
    results = []
    
    for _ in range(t):
        n = int(data[index])
        index += 1
        digits = list(map(int, data[index:index + n]))
        index += n
        
        # Calculate the initial product
        initial_product = 1
        for digit in digits:
            initial_product *= digit
        
        max_product = initial_product
        
        # Try incrementing each digit by 1 and calculate the new product
        for i in range(n):
            if digits[i] == 0:
                # If the digit is 0, the initial product is 0, so we need to handle this separately
                new_product = initial_product
                for j in range(n):
                    if j == i:
                        new_product *= (digits[j] + 1)
                    else:
                        new_product *= digits[j]
            else:
                # Calculate the new product if we increment digits[i] by 1
                new_product = initial_product // digits[i] * (digits[i] + 1)
            
            # Update the maximum product found
            if new_product > max_product:
                max_product = new_product
        
        results.append(max_product)
    
    for result in results:
        print(result)