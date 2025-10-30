def main():
    import sys
    input_data = sys.stdin.read().split()
    it = iter(input_data)
    
    # Read the first three integers: N, S, K
    N = int(next(it))
    S = int(next(it))
    K = int(next(it))
    
    total_cost = 0
    # For each product type, add its cost (price * quantity)
    for _ in range(N):
        price = int(next(it))
        quantity = int(next(it))
        total_cost += price * quantity
    
    # If total cost is less than S, add the shipping fee K
    if total_cost < S:
        total_cost += K
    
    # Print the final amount to be paid
    print(total_cost)

# Calling the main function
if __name__ == '__main__':
    main()