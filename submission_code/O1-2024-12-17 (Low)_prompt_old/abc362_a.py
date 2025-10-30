def solve():
    R, G, B = map(int, input().split())
    C = input().strip()
    
    # Based on color C, exclude its corresponding price from the set of possible prices
    valid_prices = []
    if C != 'Red':
        valid_prices.append(R)
    if C != 'Green':
        valid_prices.append(G)
    if C != 'Blue':
        valid_prices.append(B)
    
    # The answer is the minimum of remaining valid prices
    print(min(valid_prices))

# Call the solve function
solve()