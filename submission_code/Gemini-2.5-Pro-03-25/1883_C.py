# YOUR CODE HERE
import sys

# Function to compute the minimum operations for a single test case
def solve():
    # Read N and K
    n, k = map(int, sys.stdin.readline().split())
    # Read array A
    a = list(map(int, sys.stdin.readline().split()))

    # Handle cases based on K
    if k == 2:
        # For k=2, we need the product to be divisible by 2.
        # This requires at least one element to be even.
        # We find the minimum operations to make any element even.
        min_ops = float('inf')
        # Iterate through array elements
        for x in a:
            # Calculate operations needed to make x divisible by 2
            # This is 0 if x is even, 1 if x is odd.
            # The formula (k - (x % k)) % k gives the required operations.
            ops = (2 - (x % 2)) % 2 
            # Update minimum operations
            min_ops = min(min_ops, ops)
            # Optimization: if min_ops is 0, we found an even number, 
            # the product is already divisible by 2. No operations needed.
            if min_ops == 0:
                 break
        print(min_ops)

    elif k == 3:
        # For k=3, we need the product to be divisible by 3.
        # This requires at least one element to be divisible by 3.
        # We find the minimum operations to make any element divisible by 3.
        min_ops = float('inf')
        # Iterate through array elements
        for x in a:
            # Calculate operations needed to make x divisible by 3
            ops = (3 - (x % 3)) % 3
            # Update minimum operations
            min_ops = min(min_ops, ops)
            # Optimization: if min_ops is 0, we found a multiple of 3,
            # the product is already divisible by 3. No operations needed.
            if min_ops == 0:
                 break
        print(min_ops)

    elif k == 5:
        # For k=5, we need the product to be divisible by 5.
        # This requires at least one element to be divisible by 5.
        # We find the minimum operations to make any element divisible by 5.
        min_ops = float('inf')
        # Iterate through array elements
        for x in a:
            # Calculate operations needed to make x divisible by 5
            ops = (5 - (x % 5)) % 5
            # Update minimum operations
            min_ops = min(min_ops, ops)
             # Optimization: if min_ops is 0, we found a multiple of 5,
             # the product is already divisible by 5. No operations needed.
            if min_ops == 0:
                 break
        print(min_ops)

    elif k == 4:
        # For k=4, product needs to be divisible by 4. This requires either:
        # Strategy 1: At least one element divisible by 4.
        # Strategy 2: At least two elements divisible by 2. (product includes factor 2*2=4)
        # We calculate the minimum operations needed for each strategy and take the overall minimum.

        # Strategy 1: Minimum operations to make one element divisible by 4
        min_ops_one_div_4 = float('inf')
        # Count of even numbers needed for Strategy 2
        even_count = 0
        
        # Iterate through array elements to calculate values for both strategies
        for x in a:
            # Calculate operations needed to make x divisible by 4 for Strategy 1
            ops_div_4 = (4 - (x % 4)) % 4
            # Update minimum operations for strategy 1
            min_ops_one_div_4 = min(min_ops_one_div_4, ops_div_4)
            
            # Check if element is even for Strategy 2
            if x % 2 == 0:
                even_count += 1

        # Strategy 2: Minimum operations to make product divisible by 4 using factors of 2
        # Calculate minimum operations required based on initial count of even numbers
        min_ops_two_div_2 = float('inf') # Initialize with infinity
        
        if even_count >= 2:
             # Already have >= 2 even numbers, product is divisible by 4. Needs 0 ops.
             min_ops_two_div_2 = 0
        elif even_count == 1:
             # Have 1 even number. Need 1 more factor of 2 to get 2 factors total.
             # Pick an odd number (at least one must exist since N>=2) and increment once to make it even. Cost 1 op.
             min_ops_two_div_2 = 1
        else: # even_count == 0
             # Have 0 even numbers. Need 2 factors of 2.
             # Pick two distinct odd numbers (possible since N >= 2) and increment each once. Cost 1+1=2 ops.
             min_ops_two_div_2 = 2

        # The final answer is the minimum cost between the two strategies
        print(min(min_ops_one_div_4, min_ops_two_div_2))

# Read number of test cases
t = int(sys.stdin.readline())
# Process each test case
for _ in range(t):
    solve()