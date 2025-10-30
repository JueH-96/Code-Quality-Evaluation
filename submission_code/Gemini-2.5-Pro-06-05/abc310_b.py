import sys

def solve():
    """
    Reads product data and determines if any product is strictly superior to another.
    """
    # Read the number of products, N, and the maximum function ID, M.
    # M is not used in the logic, so we can use an underscore.
    try:
        n, _ = map(int, sys.stdin.readline().split())
    except (IOError, ValueError):
        # Handle cases with no input
        return

    # Store product information in a list. Each element will be a tuple
    # containing the price and a set of its functions.
    products = []
    for _ in range(n):
        data = list(map(int, sys.stdin.readline().split()))
        price = data[0]
        # data[1] is the count of functions. data[2:] contains the function IDs.
        functions = set(data[2:])
        products.append((price, functions))

    # Iterate through all ordered pairs of products (i, j) to check if
    # product i is strictly superior to product j.
    for i in range(n):
        p_i, f_i = products[i]
        for j in range(n):
            # A product cannot be strictly superior to itself.
            if i == j:
                continue
            
            p_j, f_j = products[j]

            # A product i is strictly superior to product j if:
            # 1. P_i >= P_j
            # 2. The functions of i are a subset of the functions of j.
            # 3. At least one of the following is true:
            #    a) P_i > P_j
            #    b) The functions of i are a proper subset of the functions of j.

            # Check conditions 1 and 2.
            if p_i >= p_j and f_i.issubset(f_j):
                # If the basic requirements are met, check the "strictly" superior part (condition 3).
                # Condition 3b is equivalent to len(f_i) < len(f_j), since we already know f_i is a subset of f_j.
                if p_i > p_j or len(f_i) < len(f_j):
                    # We found a pair that satisfies all conditions.
                    print("Yes")
                    return
    
    # If the loops complete without finding any such pair, the answer is "No".
    print("No")

# Execute the main function
solve()