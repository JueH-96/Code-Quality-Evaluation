import sys

# Function to solve a single test case
def solve():
    # Read the size of the array
    # n is the number of digits (1 <= n <= 9)
    n = int(sys.stdin.readline())
    
    # Read the array elements (digits)
    # a_i are digits between 0 and 9
    a = list(map(int, sys.stdin.readline().split()))

    # Initialize the maximum product found so far.
    # Since digits are non-negative (a_i >= 0), a_i + 1 >= 1.
    # The product will always be non-negative. Initializing with 0 is safe.
    max_product = 0 

    # We need to add 1 to exactly one digit. We iterate through each possible digit to increment.
    # 'i' represents the index of the digit we choose to increment.
    for i in range(n):
        # Calculate the product for the case where the digit a[i] is incremented by 1.
        # Initialize the product for this specific case to 1.
        current_product = 1
        
        # Iterate through all digits in the array to compute the product.
        # 'j' represents the index of the current digit being considered for multiplication.
        for j in range(n):
            # If the current digit's index 'j' is the same as the index 'i' we chose to increment
            if i == j:
                # Multiply the current product by the incremented value (a[j] + 1)
                current_product *= (a[j] + 1)
            else:
                # Otherwise, multiply by the original value a[j]
                current_product *= a[j]
        
        # After calculating the product for incrementing a[i], update the
        # overall maximum product found so far if this current product is larger.
        max_product = max(max_product, current_product)

    # Print the maximum product found after trying to increment each digit once.
    print(max_product)

# Read the total number of test cases
t = int(sys.stdin.readline())
# Loop through each test case and call the solve function
for _ in range(t):
    solve()