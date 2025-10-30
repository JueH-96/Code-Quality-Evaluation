import sys

def main():
    # Read N and M: number of plates eaten, number of special colors
    # N: integer, M: integer
    n, m = map(int, sys.stdin.readline().split())

    # Read the list of colors of plates eaten (C_1 ... C_N)
    # c_list: list of N strings
    c_list = sys.stdin.readline().split()

    # Read the list of special colors (D_1 ... D_M)
    # d_list: list of M strings
    d_list = sys.stdin.readline().split()

    # Read the list of prices (P_0 P_1 ... P_M)
    # p_list: list of M+1 integers
    # P_0 is the default price.
    # P_i (for i=1..M) is the price for color D_i.
    p_list = list(map(int, sys.stdin.readline().split()))

    # Extract the default price P_0 from the price list
    default_price = p_list[0]
    
    # Create a dictionary to store the prices for the special colors.
    # The keys will be the special colors (D_i) and the values will be their corresponding prices (P_i).
    # Note that the price P_i corresponds to the color D_i in the input format.
    # In the 0-indexed p_list, P_1 is at index 1, P_2 at index 2, ..., P_M at index M.
    # In the 0-indexed d_list, D_1 is at index 0, D_2 at index 1, ..., D_M at index M-1.
    # So, the price for d_list[i] is p_list[i+1].
    special_prices = {}
    for i in range(m):
        color = d_list[i]       # The i-th special color (D_{i+1} in 1-based problem index)
        price = p_list[i+1]     # The corresponding price (P_{i+1} in 1-based problem index)
        special_prices[color] = price

    # Initialize the total amount to 0
    total_amount = 0

    # Iterate through each color of the plate that Takahashi ate
    for eaten_color in c_list:
        # Determine the price for the current plate color.
        # Use the dictionary's get() method which allows specifying a default value.
        # If eaten_color exists as a key in special_prices, its value (the special price) is returned.
        # If eaten_color does not exist in special_prices, the default_price (P_0) is returned.
        price = special_prices.get(eaten_color, default_price)
        
        # Add the found price to the total amount
        total_amount += price

    # Print the final calculated total amount
    print(total_amount)

# Standard Python entry point check. Ensures main() runs when the script is executed.
if __name__ == '__main__':
    main()