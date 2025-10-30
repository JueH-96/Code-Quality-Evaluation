# YOUR CODE HERE
def solve():
    # Read N and M
    N, M = map(int, input().split())

    # Read eaten plate colors (list C)
    eaten_colors = input().split()

    # Read special dish colors (list D)
    special_dish_colors = input().split()

    # Read prices (P_0, P_1, ..., P_M)
    all_prices = list(map(int, input().split()))

    # Extract default price P_0
    # all_prices[0] corresponds to P_0
    default_price = all_prices[0]

    # Create a dictionary mapping special dish colors to their prices
    # special_dish_colors[i] is the (i+1)-th special color D_{i+1}
    # all_prices[i+1] is its price P_{i+1}
    # The loop for i from 0 to M-1 covers all M special colors.
    special_price_map = {}
    for i in range(M):
        color_name = special_dish_colors[i]
        price_value = all_prices[i+1]
        special_price_map[color_name] = price_value
    
    # Alternative using dictionary comprehension:
    # special_price_map = {special_dish_colors[i]: all_prices[i+1] for i in range(M)}

    # Calculate total price
    current_total_price = 0
    for color in eaten_colors:
        # Use the dictionary's .get() method.
        # If 'color' is in special_price_map, its value (the special price) is returned.
        # Otherwise, 'default_price' is returned.
        price_for_this_plate = special_price_map.get(color, default_price)
        current_total_price += price_for_this_plate
            
    # Print the final total price
    print(current_total_price)

if __name__ == '__main__':
    solve()