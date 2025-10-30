# YOUR CODE HERE
def calculate_total_payment():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    S = int(data[1])
    K = int(data[2])
    
    total_price = 0
    index = 3
    for _ in range(N):
        P_i = int(data[index])
        Q_i = int(data[index + 1])
        total_price += P_i * Q_i
        index += 2
    
    if total_price < S:
        total_price += K
    
    print(total_price)

calculate_total_payment()