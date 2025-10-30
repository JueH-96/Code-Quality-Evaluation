import sys

def calculate_total_price():
    # Read input from stdin
    N, M, P = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))

    # Initialize total price
    total_price = 0

    # Calculate the price of each set meal and add it to the total price
    for a in A:
        for b in B:
            total_price += min(a + b, P)

    # Print the total price
    print(total_price)

# Call the function to calculate the total price
calculate_total_price()