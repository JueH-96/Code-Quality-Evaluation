# YOUR CODE HERE
def calculate_total_price():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    # Read N and M
    N = int(data[0])
    M = int(data[1])
    
    # Read C_i
    C = data[2:2+N]
    
    # Read D_i
    D = data[2+N:2+N+M]
    
    # Read P_0 and P_i
    P = list(map(int, data[2+N+M:2+N+M+M+1]))
    
    # Create a dictionary for D_i to P_i mapping
    price_map = {D[i]: P[i+1] for i in range(M)}
    
    # Calculate the total price
    total_price = 0
    for color in C:
        if color in price_map:
            total_price += price_map[color]
        else:
            total_price += P[0]
    
    # Print the total price
    print(total_price)

calculate_total_price()