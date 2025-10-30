import sys

def solve():
    s = sys.stdin.readline().strip()
    
    # The first character is the first number (as a string)
    num1_str = s[0]
    
    # The third character is the second number (as a string)
    num2_str = s[2]
    
    # Convert the string representations to integers
    num1 = int(num1_str)
    num2 = int(num2_str)
    
    # Calculate the product
    product = num1 * num2
    
    # Print the result
    print(product)

# Call the solve function to execute the logic
solve()