import sys

def solve():
    # Read N and M from the first line of input
    N, M = map(int, sys.stdin.readline().split())

    products = []
    # Loop N times to read the details for each product
    for _ in range(N):
        # Read the entire line, parse it into integers
        line_parts = list(map(int, sys.stdin.readline().split()))
        
        # The first element is the price (P_i)
        price = line_parts[0]
        
        # The functions start from the 3rd element (index 2) onwards.
        # Convert the list of function IDs into a set for efficient lookups and comparisons.
        functions = set(line_parts[2:]) 
        
        # Store the product as a tuple of (price, functions_set)
        products.append((price, functions))

    # Flag to track if we find any pair satisfying the conditions
    found_superior = False

    # Iterate through all possible pairs of products (i, j)
    # We check if product j is strictly superior to product i.
    for i in range(N):
        for j in range(N):
            # A product cannot be strictly superior to itself
            if i == j:
                continue

            # Extract price and functions for product i
            pi, fi = products[i]
            # Extract price and functions for product j
            pj, fj = products[j]

            # Condition 1: P_i >= P_j
            # Price of product i must be greater than or equal to price of product j
            cond1 = (pi >= pj)

            # Condition 2: The j-th product has all functions of the i-th product.
            # This means the set of functions of product i (fi) must be a subset of
            # the set of functions of product j (fj).
            cond2 = fi.issubset(fj)

            # Condition 3: P_i > P_j, OR the j-th product has one or more functions
            # that the i-th product lacks.
            # "One or more functions that the i-th product lacks" means fi is a proper subset of fj.
            # This is equivalent to (fi.issubset(fj) AND fi != fj).
            # Since cond2 already ensures fi.issubset(fj), we only need to check fi != fj here for that part.
            cond3 = (pi > pj) or (fi != fj)

            # If all three conditions are met, we have found a strictly superior pair
            if cond1 and cond2 and cond3:
                found_superior = True
                break # Exit the inner loop since we found a pair
        
        if found_superior:
            break # Exit the outer loop since we found a pair

    # Print the final result
    if found_superior:
        print("Yes")
    else:
        print("No")

# Call the solve function to run the program
solve()