import sys

def solve():
    N, K = map(int, sys.stdin.readline().split())
    cakes = []
    for _ in range(N):
        x, y, z = map(int, sys.stdin.readline().split())
        cakes.append((x, y, z))

    max_total_price = 0

    # Iterate through all 8 combinations of signs for X, Y, Z
    # sx, sy, sz will be -1 or 1
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                # Calculate a "score" for each cake based on current sign combination
                # and store (score, original_index)
                scored_cakes = []
                for i in range(N):
                    x, y, z = cakes[i]
                    score = sx * x + sy * y + sz * z
                    scored_cakes.append((score, i))
                
                # Sort cakes by their score in descending order
                # This ensures we pick the "best" 2K cakes according to this score
                scored_cakes.sort(key=lambda item: item[0], reverse=True)
                
                current_total_price = 0
                # Take the top 2K cakes and form K pairs.
                # The greedy pairing (1st with 2nd, 3rd with 4th, etc.) is known to work
                # for this type of problem structure.
                for j in range(K):
                    # Get the original indices of the two cakes forming the pair
                    cake_idx1 = scored_cakes[2 * j][1]
                    cake_idx2 = scored_cakes[2 * j + 1][1]

                    # Get the actual (X, Y, Z) values for the cakes using their original indices
                    c1_x, c1_y, c1_z = cakes[cake_idx1]
                    c2_x, c2_y, c2_z = cakes[cake_idx2]

                    # Calculate the price of this pair using the problem's definition
                    price_pair = max(c1_x + c2_x, c1_y + c2_y, c1_z + c2_z)
                    
                    current_total_price += price_pair
                
                # Update the overall maximum total price found so far
                max_total_price = max(max_total_price, current_total_price)
    
    # Print the result for the current test case
    sys.stdout.write(str(max_total_price) + "
")

# Read the number of test cases
T = int(sys.stdin.readline())
# Solve each test case
for _ in range(T):
    solve()