# YOUR CODE HERE
import sys

# Function to solve the problem
def solve():
    # Read N and M
    N, M = map(int, sys.stdin.readline().split())

    # Store product data as a list of tuples: (price, set_of_functions)
    products = []
    # Read N products
    for _ in range(N):
        line = list(map(int, sys.stdin.readline().split()))
        P = line[0]
        C = line[1]
        # The remaining elements are functions F_i,j. Convert them to a set for efficient checks.
        # Note: line[2:] gives a list of integers. set() converts it to a set.
        F = set(line[2:])
        products.append((P, F))

    # Flag to indicate if a pair (i, j) is found such that j is strictly superior to i
    found_superior = False

    # Iterate through all possible pairs (i, j) of products (0-indexed)
    # We check if product j is strictly superior to product i
    for i in range(N):
        for j in range(N):
            # Get product i and product j data
            # We are checking if product `j` is strictly superior to product `i`
            Pi, Si = products[i]
            Pj, Sj = products[j]

            # Conditions for product j to be strictly superior to product i:
            # 1. Pi >= Pj (Product j is no more expensive than product i)
            # 2. Si is a subset of Sj (Product j has all functions of product i)
            # 3. Pi > Pj OR Si is a strict subset of Sj (Product j is strictly better)
            #    A strict subset check (Si < Sj) is equivalent to Si <= Sj AND Si != Sj.
            #    Since we already checked Si <= Sj in condition 2 and it passed,
            #    condition 3 simplifies to Pi > Pj OR Si != Sj.

            # Check condition 1: Pi >= Pj
            if Pi < Pj:
                continue # Product j is more expensive than product i, cannot be superior to i

            # Check condition 2: Si is a subset of Sj (j has all functions of i)
            # This means all functions in Si must be present in Sj.
            if not Si.issubset(Sj):
                continue # Product j lacks some function present in product i, cannot be superior to i

            # Check condition 3: Pi > Pj OR Si != Sj (j is strictly better than i)
            # This condition ensures there is a strict advantage of j over i.
            # If Pi == Pj and Si == Sj, this condition (Pi > Pj or Si != Sj) becomes (False or False) which is False.
            # This correctly handles the case where i and j are identical or have identical specs - neither is strictly superior to the other.
            if Pi > Pj or Si != Sj:
                # All conditions met: product j is strictly superior to product i
                found_superior = True
                # Found a pair (i, j) where j is strictly superior to i, we can stop searching
                break # Break the inner loop (checking different j for current i)
        if found_superior:
            break # Break the outer loop as well (checking different i)

    # Print the result
    print("Yes" if found_superior else "No")

# Call the solve function to run the program logic
solve()
# END YOUR CODE HERE