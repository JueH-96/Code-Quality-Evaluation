def max_pair_price(a, b):
    # Returns the price of pairing cake a with cake b
    return max(a[0] + b[0], a[1] + b[1], a[2] + b[2])

def solve_case():
    # Read N and K
    N, K = map(int, input().split())
    
    # Read cakes data
    cakes = []
    for _ in range(N):
        x, y, z = map(int, input().split())
        cakes.append((x, y, z))
    
    # Try all possible pairs and store their prices
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            price = max_pair_price(cakes[i], cakes[j])
            pairs.append((price, i, j))
    
    # Sort pairs by price in descending order
    pairs.sort(reverse=True)
    
    # Greedily select K pairs ensuring no cake is used twice
    used = set()
    total_price = 0
    pairs_selected = 0
    
    for price, i, j in pairs:
        if pairs_selected == K:
            break
        if i not in used and j not in used:
            total_price += price
            used.add(i)
            used.add(j)
            pairs_selected += 1
    
    return total_price

# Read number of test cases
T = int(input())

# Process each test case
for _ in range(T):
    result = solve_case()
    print(result)