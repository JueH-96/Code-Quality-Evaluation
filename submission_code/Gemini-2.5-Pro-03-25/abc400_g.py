# YOUR CODE HERE
import sys

# Function to compute the answer for a single test case
def solve():
    # Read N (number of cakes) and K (number of pairs)
    # N is the total number of kinds of cakes available.
    # K is the number of pairs we need to form.
    N, K = map(int, sys.stdin.readline().split())
    
    # Read cake properties (beauty X_i, tastiness Y_i, popularity Z_i) for each cake i
    cakes = []
    for _ in range(N):
        # Store cake data as a list [X, Y, Z]
        cakes.append(list(map(int, sys.stdin.readline().split()))) 

    # Initialize the variable to store the maximum possible total price found so far.
    # The price of a pair is defined as max(X_a+X_b, Y_a+Y_b, Z_a+Z_b).
    # Since X_i, Y_i, Z_i are non-negative, the sums X_a+X_b, etc., are non-negative.
    # The maximum of non-negative values is non-negative.
    # The total price is a sum of K non-negative pair prices, so it must be non-negative.
    # We can safely initialize max_total_price with 0.
    max_total_price = 0 

    # This problem structure, maximizing a sum of maximums of linear combinations,
    # often relates to a technique where we consider maximizing linear combinations
    # with different signs. There are 3 attributes (X, Y, Z), so we consider combinations
    # s1*X + s2*Y + s3*Z where s1, s2, s3 can be +1 or -1.
    # There are 2^3 = 8 such sign combinations.
    
    # Generate all 8 sign combinations for (s1, s2, s3)
    signs = []
    for s1 in [-1, 1]:
        for s2 in [-1, 1]:
            for s3 in [-1, 1]:
                signs.append((s1, s2, s3))

    # Iterate through all 8 sign combinations
    for s1, s2, s3 in signs:
        # For the current sign combination (s1, s2, s3), calculate the value 
        # V_j = s1*X_j + s2*Y_j + s3*Z_j for each cake j.
        # This value represents the projection of the cake's attribute vector (X_j, Y_j, Z_j)
        # onto the direction vector (s1, s2, s3).
        V_values = []
        for i in range(N):
            X, Y, Z = cakes[i]
            V_values.append(s1 * X + s2 * Y + s3 * Z)
        
        # To maximize the sum of V_j over 2K chosen cakes, we should pick the 2K cakes
        # with the largest V_j values. We sort the calculated V_values in descending order.
        V_values.sort(reverse=True)
        
        # The technique posits that the maximum total price is the maximum value obtained
        # by summing the V_j values for the top 2K cakes, across all 8 sign combinations.
        # We select the top 2K cakes (since we form K pairs, using 2K cakes in total).
        # Calculate the sum of V_values for these top 2K cakes.
        current_total_price = sum(V_values[:K*2]) # Sum the largest 2K values
        
        # Update the overall maximum total price found across all sign combinations.
        # This maximum value is the answer to the problem.
        max_total_price = max(max_total_price, current_total_price)

    # Print the final maximum total price computed for this test case
    print(max_total_price)

# Read the number of test cases from standard input
T = int(sys.stdin.readline())
# Process each test case by calling the solve function
for _ in range(T):
    solve()