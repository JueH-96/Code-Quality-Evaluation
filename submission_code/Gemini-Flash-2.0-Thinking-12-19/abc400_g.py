import sys

def solve():
    """Reads a single test case and prints the maximum total price."""
    N, K = map(int, sys.stdin.readline().split())
    cakes = []
    for _ in range(N):
        X, Y, Z = map(int, sys.stdin.readline().split())
        cakes.append((X, Y, Z))

    max_total_price = 0

    # The 8 combinations of signs for X, Y, and Z attributes.
    # These correspond to the directions (+X+Y+Z), (+X+Y-Z), etc.
    signs = [(1, 1, 1), (1, 1, -1), (1, -1, 1), (-1, 1, 1),
             (1, -1, -1), (-1, 1, -1), (-1, -1, 1), (-1, -1, -1)]

    # Iterate through each sign combination
    for p, q, r in signs:
        # Calculate a score for each cake based on the current sign combination.
        # The score is p*X_i + q*Y_i + r*Z_i.
        # We store (score, original_index) pairs.
        scored_cakes = []
        for i in range(N):
            score = p * cakes[i][0] + q * cakes[i][1] + r * cakes[i][2]
            scored_cakes.append((score, i))

        # Sort cakes by their calculated score in descending order.
        # This brings the cakes with the highest scores for this combination to the front.
        scored_cakes.sort(key=lambda item: item[0], reverse=True)

        # Select the top 2K cakes based on the sorted scores.
        # We extract the original indices of these top cakes.
        top_cakes_indices_sorted = [item[1] for item in scored_cakes[:2*K]]

        # Calculate the total price for the K pairs formed by these top cakes.
        # The pairing strategy is simple: pair the 1st with the 2nd, 3rd with 4th, ..., (2K-1)th with 2Kth
        # based on their order in the sorted list of top 2K cakes.
        current_total_price = 0
        for m in range(K):
            # Get the original indices of the two cakes in the current pair
            idx1 = top_cakes_indices_sorted[2*m]
            idx2 = top_cakes_indices_sorted[2*m + 1]

            # Get the attributes (X, Y, Z) of the cakes using their original indices
            x1, y1, z1 = cakes[idx1]
            x2, y2, z2 = cakes[idx2]

            # Calculate the price of this pair
            pair_price = max(x1 + x2, y1 + y2, z1 + z2)
            current_total_price += pair_price

        # Update the maximum total price found so far across all sign combinations.
        max_total_price = max(max_total_price, current_total_price)

    # Print the final maximum total price.
    print(max_total_price)


# Read the total number of test cases.
T = int(sys.stdin.readline())

# Process each test case by calling the solve function.
for _ in range(T):
    solve()