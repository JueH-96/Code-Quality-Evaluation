import math

def solve():
    n = int(input())
    a = list(map(int, input().split()))

    max_product = 0

    # Iterate through each digit in the array
    # For each digit, consider incrementing it by 1
    for i in range(n):
        # Create a temporary copy of the original array
        # This is important so that the original 'a' is not modified
        # for subsequent iterations of the outer loop.
        temp_a = list(a) 
        
        # Increment the digit at the current index 'i' by 1
        temp_a[i] += 1
        
        # Calculate the product of all elements in the modified temporary array
        current_product = math.prod(temp_a) # math.prod is efficient for this
            
        # Update the maximum product found so far
        max_product = max(max_product, current_product)
            
    # Print the final maximum product for the current test case
    print(max_product)

# Read the number of test cases
t = int(input())

# Loop through each test case and call the solve function
for _ in range(t):
    solve()