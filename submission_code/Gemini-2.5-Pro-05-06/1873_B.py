def solve_single_case():
    n = int(input())  # Read the number of digits
    a = list(map(int, input().split()))  # Read the digits into a list

    max_product_achieved = 0  # Initialize with 0, as products of digits (0-9) are non-negative

    # Iterate through each index 'i'. a[i] will be the digit to increment.
    for i in range(n):
        current_product = 1  # Initialize product for this specific modification
        
        # Calculate the product if a[i] is incremented
        # Iterate through all digits (indexed by 'j') to compute the product
        for j in range(n):
            if i == j:  # If this is the digit we chose to increment
                current_product *= (a[j] + 1)
            else:  # For all other digits, use their original value
                current_product *= a[j]
        
        # Update the overall maximum product found so far
        if current_product > max_product_achieved:
            max_product_achieved = current_product
            
    print(max_product_achieved)  # Print the result for this test case

# Read the total number of test cases
num_test_cases = int(input())

# Loop through each test case and solve it
for _ in range(num_test_cases):
    solve_single_case()