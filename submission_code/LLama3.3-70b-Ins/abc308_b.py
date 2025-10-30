import sys

def calculate_total_price():
    # Read the number of plates and colors
    N, M = map(int, sys.stdin.readline().split())

    # Read the colors of the plates
    C = sys.stdin.readline().split()

    # Read the distinct colors and their prices
    D = sys.stdin.readline().split()
    P = list(map(int, sys.stdin.readline().split()))

    # Initialize the total price
    total_price = 0

    # Iterate over each plate
    for color in C:
        # Check if the color is in the distinct colors
        if color in D:
            # Add the price of the color to the total price
            total_price += P[D.index(color) + 1]
        else:
            # Add the default price to the total price
            total_price += P[0]

    # Print the total price
    print(total_price)

# Call the function
calculate_total_price()