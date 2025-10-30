# YOUR CODE HERE
T = int(input())

for _ in range(T):
    N, K = map(int, input().split())
    
    cakes = []
    for _ in range(N):
        x, y, z = map(int, input().split())
        cakes.append((x, y, z))
    
    # Generate all possible pairs with their prices
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            x1, y1, z1 = cakes[i]
            x2, y2, z2 = cakes[j]
            price = max(x1 + x2, y1 + y2, z1 + z2)
            pairs.append((price, i, j))
    
    # Sort pairs by price in descending order
    pairs.sort(reverse=True)
    
    # Greedily select K pairs
    used = [False] * N
    total_price = 0
    selected_pairs = 0
    
    for price, i, j in pairs:
        if not used[i] and not used[j]:
            used[i] = True
            used[j] = True
            total_price += price
            selected_pairs += 1
            if selected_pairs == K:
                break
    
    print(total_price)