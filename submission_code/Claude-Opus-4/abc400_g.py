# YOUR CODE HERE
def solve_case():
    n, k = map(int, input().split())
    cakes = []
    for _ in range(n):
        x, y, z = map(int, input().split())
        cakes.append((x, y, z))
    
    # Generate all possible pairs with their prices
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1, z1 = cakes[i]
            x2, y2, z2 = cakes[j]
            price = max(x1 + x2, y1 + y2, z1 + z2)
            pairs.append((price, i, j))
    
    # Sort pairs by price in descending order
    pairs.sort(reverse=True)
    
    # Greedily select k pairs
    used = [False] * n
    total_price = 0
    pairs_selected = 0
    
    for price, i, j in pairs:
        if not used[i] and not used[j]:
            used[i] = True
            used[j] = True
            total_price += price
            pairs_selected += 1
            if pairs_selected == k:
                break
    
    return total_price

# Read number of test cases
t = int(input())

# Process each test case
for _ in range(t):
    result = solve_case()
    print(result)