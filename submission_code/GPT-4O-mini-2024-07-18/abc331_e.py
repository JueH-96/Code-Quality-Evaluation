def most_expensive_set_meal():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N, M, L
    N, M, L = map(int, data[0].split())
    
    # Read main dish prices
    a = list(map(int, data[1].split()))
    
    # Read side dish prices
    b = list(map(int, data[2].split()))
    
    # Read the pairs that are not offered
    forbidden_pairs = set()
    for i in range(3, 3 + L):
        c, d = map(int, data[i].split())
        forbidden_pairs.add((c - 1, d - 1))  # Store as 0-indexed
    
    # Initialize the maximum price
    max_price = 0
    
    # Calculate the price of each valid set meal
    for i in range(N):
        for j in range(M):
            if (i, j) not in forbidden_pairs:
                price = a[i] + b[j]
                if price > max_price:
                    max_price = price
    
    # Print the result
    print(max_price)

most_expensive_set_meal()