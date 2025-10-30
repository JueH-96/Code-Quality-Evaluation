# YOUR CODE HERE
def calculate_total_sushi_price():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read N and M
    N, M = map(int, data[0].split())
    
    # Read the colors of the plates
    C = data[1].split()
    
    # Read the distinct colors and their prices
    D = data[2].split()
    P = list(map(int, data[3].split()))
    
    # Create a dictionary to map colors to their prices
    price_map = {D[i]: P[i + 1] for i in range(M)}
    price_map['default'] = P[0]  # P_0 for colors not in D
    
    # Calculate the total price
    total_price = 0
    for color in C:
        total_price += price_map.get(color, price_map['default'])
    
    print(total_price)

calculate_total_sushi_price()