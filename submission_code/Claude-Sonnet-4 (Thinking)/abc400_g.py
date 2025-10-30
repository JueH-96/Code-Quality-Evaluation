def solve():
    n, k = map(int, input().split())
    cakes = []
    for i in range(n):
        x, y, z = map(int, input().split())
        cakes.append((x, y, z))
    
    # Calculate all possible pair prices
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            x_i, y_i, z_i = cakes[i]
            x_j, y_j, z_j = cakes[j]
            price = max(x_i + x_j, y_i + y_j, z_i + z_j)
            pairs.append((price, i, j))
    
    # Sort pairs by price in descending order
    pairs.sort(reverse=True)
    
    # Greedily select pairs
    used = set()
    total_price = 0
    selected_pairs = 0
    
    for price, i, j in pairs:
        if i not in used and j not in used:
            used.add(i)
            used.add(j)
            total_price += price
            selected_pairs += 1
            if selected_pairs == k:
                break
    
    return total_price

t = int(input())
for _ in range(t):
    print(solve())