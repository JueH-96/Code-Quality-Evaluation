import sys

def solve():
    # Read N and M from input: N is number of products, M is max function ID
    N, M = map(int, sys.stdin.readline().split())

    # List to store product information
    products = []
    # Read N product specifications
    for _ in range(N):
        # Read line containing price, function count, and function IDs
        # Example line: P C F1 F2 ... FC
        line = list(map(int, sys.stdin.readline().split()))
        P = line[0]  # Price of the product
        # C = line[1] # The number of functions C is line[1], but it's implicitly len(line[2:])
        # Store function IDs in a set for efficient subset checking.
        # The function IDs start from index 2 of the line list.
        F = set(line[2:]) # Set of function IDs for the product
        
        # Store product information (price and set of functions) as a dictionary
        products.append({'price': P, 'functions': F})

    # Flag to indicate if a pair (i, j) satisfying the conditions is found
    found = False

    # Iterate through all possible pairs of products (i, j)
    # The indices i and j range from 0 to N-1.
    for i in range(N):
        for j in range(N):
            # A product cannot be strictly superior to itself, so technically we could skip i == j.
            # However, the conditions naturally handle this: if i == j, P_i == P_j and S_i == S_j.
            # Then condition 1 (Pi >= Pj) is true, condition 2 (Si subset Sj) is true.
            # Condition 3 (Pi > Pj or Si proper subset Sj) becomes (False or False) = False.
            # So the pair (i, i) will never satisfy all conditions. We don't strictly need to skip i == j.
            
            # Extract information for product i
            Pi = products[i]['price']
            Si = products[i]['functions']
            
            # Extract information for product j
            Pj = products[j]['price']
            Sj = products[j]['functions']

            # Check the three conditions defined in the problem statement for product i being strictly superior to product j:

            # Condition 1: Price of product i is greater than or equal to the price of product j (P_i >= P_j)
            if Pi >= Pj:
                # Condition 2: Product j has all the functions of product i (S_i is a subset of S_j)
                # Use the `issubset` method of Python sets for efficiency.
                if Si.issubset(Sj):
                    # Condition 3: Either the price of product i is strictly greater than the price of product j (P_i > P_j),
                    # OR product j has at least one function that product i does not have.
                    # The second part ("product j has functions i lacks") combined with Condition 2 (Si subset Sj)
                    # means that Si must be a *proper* subset of Sj (S_i != S_j).
                    
                    # Check if P_i > P_j
                    condition3_part1 = (Pi > Pj)
                    
                    # Check if Si is a proper subset of Sj. Since we already know Si.issubset(Sj) is true here,
                    # Si is a proper subset of Sj if and only if Si is not equal to Sj.
                    # This can be checked efficiently by comparing the number of functions (sizes of the sets).
                    # If len(Si) < len(Sj), then Si must be a proper subset of Sj.
                    condition3_part2 = (len(Si) < len(Sj))
                    
                    # If either part of condition 3 is true, then the overall condition 3 holds.
                    if condition3_part1 or condition3_part2:
                        # If we reach here, all three conditions are met for the pair (i, j).
                        found = True
                        # We found a pair satisfying the conditions, so we can stop searching.
                        break  # Exit the inner loop (over j)
        
        # If a pair was found in the inner loop (found = True), exit the outer loop (over i) as well.
        if found:
            break

    # After checking all pairs (or stopping early), print the final result.
    if found:
        print("Yes")
    else:
        # If the loops completed without finding such a pair
        print("No")

# Execute the solve function when the script is run directly.
if __name__ == '__main__':
    solve()