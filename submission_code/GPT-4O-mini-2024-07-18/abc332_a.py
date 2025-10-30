# YOUR CODE HERE
def calculate_total_payment():
    import sys
    input = sys.stdin.read
    data = input().splitlines()
    
    # Read the first line for N, S, K
    N, S, K = map(int, data[0].split())
    
    total_price = 0
    
    # Read the next N lines for P_i and Q_i
    for i in range(1, N + 1):
        P_i, Q_i = map(int, data[i].split())
        total_price += P_i * Q_i
    
    # Determine the shipping fee
    shipping_fee = 0 if total_price >= S else K
    
    # Calculate the total amount to pay
    total_amount = total_price + shipping_fee
    
    # Print the result
    print(total_amount)

calculate_total_payment()